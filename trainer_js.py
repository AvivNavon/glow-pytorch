import logging

import numpy as np
from tqdm import tqdm, trange
from math import log
from pathlib import Path

import argparse

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms, utils
from utils import common_parser, set_seed, set_logger
import wandb

from model import Glow
# from data import load_dataset_with_kl
from data import load_dataset_with_kl

from imagegpt.imagegpt import ImageGPT

parser = argparse.ArgumentParser(description="Glow trainer", parents=[common_parser])
parser.add_argument("--batch-size", default=16, type=int, help="batch size")
parser.add_argument("--epochs", default=15, type=int, help="number of epochs")
# parser.add_argument("--iters", default=50000, type=int, help="number of training iterations")
parser.add_argument(
    "--n_flow", default=32, type=int, help="number of flows in each block"
)
parser.add_argument("--n_block", default=4, type=int, help="number of blocks")
parser.add_argument(
    "--no_lu",
    action="store_true",
    help="use plain convolution instead of LU decomposed version",
)
parser.add_argument(
    "--affine", action="store_true", help="use affine coupling instead of additive"
)
# parser.add_argument(
#     '--loss', default='kl', type=str, choices=['kl', 'reverse-kl'], help="KL type for loss: ['kl', 'reverse-kl']"
# )
parser.add_argument("--n_bits", default=5, type=int, help="number of bits")
parser.add_argument("--samples-every", default=250, type=int, help="samples every")
parser.add_argument("--model-every", default=25000, type=int, help="model every")
parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
parser.add_argument(
    "--unif-prob", default=1e-2, type=float, help="probability for calculating log-probs from uniform dist."
)
parser.add_argument("--model-ll-clamp", default=-np.inf, type=float, help="model log likelihood clamp")
parser.add_argument("--clip", default=100, type=float, help="grad clipping")
parser.add_argument("--img_size", default=32, type=int, help="image size")
parser.add_argument("--workers", default=0, type=int, help="num workers")
parser.add_argument("--temp", default=0.7, type=float, help="temperature of sampling")
parser.add_argument("--n_sample", default=20, type=int, help="number of samples")
parser.add_argument("--sample-path", type=str, default='samples/forward_reverse_kl', help="Path to image directory")
parser.add_argument("--path-to-clusters", type=Path, help="Path to image directory", default='./data/kmeans_centers.npy')
parser.add_argument('--model-path', default='checkpoint', type=Path)
parser.add_argument("--path", default='data', type=Path, help="Path to image directory")
parser.add_argument("--seed", default=42, type=int, help="random seed")
# image gpt
parser.add_argument('--n-gpu', default=1, type=int)
parser.add_argument("--tf-device", nargs="+", type=int, default=[0], help="GPU devices for tf")
parser.add_argument("--pt-device", nargs="+", type=int, default=[0], help="GPU devices for pt")
parser.add_argument('--imagegpt-artifact', default='../image-gpt/artifacts', type=Path)


def get_loader(path, clusters_path, sample_flag=False, device=None, batch_size=16):
    # todo: refactor
    # todo: we need to add augmentations like in train.py
    train, test = load_dataset_with_kl(path=path, clusters_path=clusters_path, sample_flag=sample_flag, device=device)
    dataset = TensorDataset(*train)
    test_dataset = TensorDataset(*test)

    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size, num_workers=args.workers)

    return loader, test_loader


def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


def train(args, model, optimizer, image_gpt: ImageGPT):
    sample_path = Path(args.sample_path)
    sample_path.mkdir(exist_ok=True, parents=True)

    model_path = args.model_path / f'forward_reverse_kl'
    model_path.mkdir(exist_ok=True, parents=True)

    # loss
    n_pixel = args.img_size * args.img_size * 3

    def calc_loss(log_p, log_q, logdet, forward_log_p, forward_logdet, ll_generated_data):
        """Reverse KL

        :param log_p: data log likelihood
        :param log_q: log q (model)
        :param logdet: log determinant

        :return: loss
        """
        # log likelihood of model glow on z
        model_ll_z = (logdet + log_q - log(n_bins) * n_pixel) / (n_pixel * log(2))
        # log likelihood data (imageGPT) on z
        data_ll_z = log_p

        # log likelihood of model glow on x
        model_ll_x = (forward_logdet + forward_log_p - log(n_bins) * n_pixel) / (n_pixel * log(2))
        # log likelihood data (imageGPT) on x
        data_ll_x = ll_generated_data.to(model_ll_x.device)

        # logM_x = log(0.5) + torch.log(torch.exp(model_ll_x.squeeze()) + torch.exp(data_ll_x.squeeze()))
        # logM_z = log(0.5) + torch.log(torch.exp(model_ll_z.squeeze()) + torch.exp(data_ll_z.squeeze()))

        logM_z = log(0.5) + torch.logsumexp(torch.stack((model_ll_z.squeeze(), data_ll_z.squeeze())), dim=0)
        logM_x = log(0.5) + torch.logsumexp(torch.stack((model_ll_x.squeeze(), data_ll_x.squeeze())), dim=0)

        return 0.5 * (
                (data_ll_x - logM_x).sum() +
                (
                        (model_ll_z - logM_z).sum()
                        # NOTE: this is from the log derivative trick
                        + ((model_ll_z - logM_z).detach() * model_ll_z).sum()
                )
        )

    n_bins = 2.0 ** args.n_bits

    def gen_batch(batch_size=args.batch_size):
        normal = torch.distributions.normal.Normal(loc=0., scale=args.temp)
        z_sample = []
        z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
        log_probs = []
        for z in z_shapes:
            z_new = torch.randn(batch_size, *z) * args.temp
            z_sample.append(z_new.to(device))
            log_probs.append(normal.log_prob(z_new).sum((1, 2, 3)))
        return z_sample, log_probs

    laoder, test_loader = get_loader(args.path, args.path_to_clusters, device=device, batch_size=args.batch_size)

    pbar = trange(args.epochs)
    global_iter = 0
    for i in pbar:
        for _, x_batch in enumerate(laoder):
            model.train()

            z_batch, log_probs = gen_batch()
            # todo: we need logdet for each z in the batch?
            sampled_images, reverse_logdet = model.module.reverse(z_batch)
            # todo: how to calc. log probs?
            log_p = sum(log_probs)
            log_p = log_p.to(device)
            sampled_images = torch.clamp(sampled_images, -.5, .5)

            # pass through Glow
            (image, nll_generated_data) = x_batch
            image = image.to(device)
            if args.n_bits < 8:
                image = torch.floor(image / 2 ** (8 - args.n_bits))

            image = image / n_bins - 0.5
            forward_log_p, forward_logdet, _ = model(image + torch.rand_like(image) / n_bins)

            # pass through image gpt
            sampled_images_numpy = sampled_images.permute(0, 2, 3, 1).detach().cpu().numpy()
            # NOTE: expect channels last
            # clusters are in (-1, 1)
            sampled_images_numpy = sampled_images_numpy * 2.
            clustered_sampled_images = image_gpt.color_quantize(sampled_images_numpy)
            data_nll = image_gpt.eval_model(clustered_sampled_images)

            # loss and metrics
            logdet = reverse_logdet.mean()  # todo: verify
            data_log_likelihood = -torch.from_numpy(np.concatenate(data_nll)).to(device)
            ll_generated_data = -nll_generated_data

            loss = calc_loss(
                log_p=data_log_likelihood, log_q=log_p, logdet=logdet,  # on z
                forward_log_p=forward_log_p, forward_logdet=forward_logdet, ll_generated_data=ll_generated_data  # on x
            )
            model_ll = ((log_p + logdet) / n_pixel).mean()
            log_p = (log_p / (log(2) * n_pixel)).mean()
            log_det = (logdet / (log(2) * n_pixel)).mean()
            data_nll = np.concatenate(data_nll).mean()
            model.zero_grad()
            loss.backward()
            # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
            warmup_lr = args.lr
            optimizer.param_groups[0]["lr"] = warmup_lr
            # clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            pbar.set_description(
                f"Loss: {loss.item():.5f}; "
                f"data LL: {-data_nll:.5f};  model LL: {model_ll.item():.5f}; "
                f"logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}"
            )

            if args.wandb:
                wandb.log({
                    'loss': loss.item(),
                    'data_ll': -data_nll,
                    'model_ll': model_ll.item(),
                    'logP': log_p.item(),
                    'logdet': log_det.item()
                })

            if global_iter % args.samples_every == 0:
                with torch.no_grad():

                    z_sample, _ = gen_batch(batch_size=args.n_sample)
                    try:
                        utils.save_image(
                            model_single.reverse(z_sample)[0].cpu().data,
                            sample_path / f"{str(global_iter + 1).zfill(6)}.png",
                            normalize=True,
                            nrow=10,
                            # range=(-0.5, 0.5),
                            value_range=(-0.5, 0.5),
                            )
                    except:
                        utils.save_image(
                            model_single.reverse(z_sample)[0].cpu().data,
                            sample_path / f"{str(global_iter + 1).zfill(6)}.png",
                            normalize=True,
                            nrow=10,
                            range=(-0.5, 0.5),
                            # value_range=(-0.5, 0.5),
                        )
                if args.wandb:
                    wandb.save((sample_path / f"{str(global_iter + 1).zfill(6)}.png").as_posix())

            if global_iter % args.model_every == 0:
                torch.save(
                    model.state_dict(), model_path / f"model_{str(global_iter + 1).zfill(6)}_lr_{args.lr}_seed_{args.seed}.pt"
                )
                torch.save(
                    optimizer.state_dict(), model_path / f"optim_{str(global_iter + 1).zfill(6)}_lr_{args.lr}_seed_{args.seed}.pt"
                )

            global_iter += 1

        # evaluate
        model.eval()
        val_loss = 0.
        total = 0.
        with torch.no_grad():
            for _, x_batch in enumerate(laoder):
                z_batch, log_probs = gen_batch()
                # todo: we need logdet for each z in the batch?
                sampled_images, reverse_logdet = model.module.reverse(z_batch)
                # todo: how to calc. log probs?
                log_p = sum(log_probs)
                log_p = log_p.to(device)
                sampled_images = torch.clamp(sampled_images, -.5, .5)

                # pass through Glow
                (image, nll_generated_data) = x_batch
                image = image.to(device)
                if args.n_bits < 8:
                    image = torch.floor(image / 2 ** (8 - args.n_bits))

                image = image / n_bins - 0.5
                forward_log_p, forward_logdet, _ = model(image + torch.rand_like(image) / n_bins)

                # pass through image gpt
                sampled_images_numpy = sampled_images.permute(0, 2, 3, 1).detach().cpu().numpy()
                # NOTE: expect channels last
                # clusters are in (-1, 1)
                sampled_images_numpy = sampled_images_numpy * 2.
                clustered_sampled_images = image_gpt.color_quantize(sampled_images_numpy)
                data_nll = image_gpt.eval_model(clustered_sampled_images)

                # loss and metrics
                logdet = reverse_logdet.mean()  # todo: verify
                data_log_likelihood = -torch.from_numpy(np.concatenate(data_nll)).to(device)
                ll_generated_data = -nll_generated_data

                loss = calc_loss(
                    log_p=data_log_likelihood, log_q=log_p, logdet=logdet,  # on z
                    forward_log_p=forward_log_p, forward_logdet=forward_logdet, ll_generated_data=ll_generated_data
                )
                val_loss += loss.item()
                total += len(data_log_likelihood)

        val_loss /= total
        if args.wandb:
            wandb.log({'val/loss': val_loss, 'epoch': i})

    torch.save(
        model.state_dict(), model_path / f"model_end_lr_{args.lr}_seed_{args.seed}.pt"
    )
    torch.save(
        optimizer.state_dict(), model_path / f"optim_end_lr_{args.lr}_seed_{args.seed}.pt"
    )


if __name__ == "__main__":
    args = parser.parse_args()
    set_logger()
    # set seed
    set_seed(args.seed)

    device = torch.device(f"cuda:{args.pt_device[0]}" if torch.cuda.is_available() else "cpu")

    # Glow
    model_single = Glow(
        3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu, reverse_log_det=True
    )
    model = nn.DataParallel(model_single, device_ids=args.pt_device)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # image gpt
    artifacts_path = args.imagegpt_artifact
    image_gpt = ImageGPT(
        batch_size=args.batch_size,
        devices=args.tf_device,
        ckpt_path=(artifacts_path / "model.ckpt-1000000").as_posix(),
        color_cluster_path=(artifacts_path / "kmeans_centers.npy").as_posix(),
    )

    # wandb
    if args.wandb:
        name = f"glow_imagegpt_JS_lr_{args.lr}_bs_{args.batch_size}_no_LU_{args.no_lu}_seed_{args.seed}"
        wandb.init(project="glow", entity='avivnav', name=name)
        wandb.config.update(args)

    train(args, model, optimizer, image_gpt)