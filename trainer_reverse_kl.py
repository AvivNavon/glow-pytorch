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
from data import load_dataset_with_kl

from imagegpt.imagegpt import ImageGPT

parser = argparse.ArgumentParser(description="Glow trainer", parents=[common_parser])
parser.add_argument("--batch-size", default=16, type=int, help="batch size")
parser.add_argument("--iters", default=50000, type=int, help="number of training iterations")
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
parser.add_argument("--model-every", default=500000, type=int, help="model every")
parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
parser.add_argument("--model-ll-clamp", default=-np.inf, type=float, help="model log likelihood clamp")
parser.add_argument("--clip", default=100, type=float, help="grad clipping")
parser.add_argument("--img_size", default=32, type=int, help="image size")
parser.add_argument("--workers", default=0, type=int, help="num workers")
parser.add_argument("--temp", default=0.7, type=float, help="temperature of sampling")
parser.add_argument("--n_sample", default=20, type=int, help="number of samples")
parser.add_argument("--sample-path", type=str, default='samples', help="Path to image directory")
parser.add_argument("--path-to-clusters", type=Path, help="Path to image directory", default='./data/kmeans_centers.npy')
parser.add_argument('--model-path', default='checkpoint', type=Path)
parser.add_argument("--seed", default=42, type=int, help="random seed")
# image gpt
parser.add_argument('--n-gpu', default=1, type=int)
parser.add_argument("--tf-device", nargs="+", type=int, default=[0], help="GPU devices for tf")
parser.add_argument("--pt-device", nargs="+", type=int, default=[0], help="GPU devices for pt")
parser.add_argument('--imagegpt-artifact', default='../image-gpt/artifacts', type=Path)


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

    model_path = args.model_path / f'reverse_kl'
    model_path.mkdir(exist_ok=True, parents=True)

    # loss
    n_pixel = args.img_size * args.img_size * 3

    def calc_loss(log_p, log_q, logdet):
        """Reverse KL

        :param log_p: data log likelihood
        :param log_q: log q (model)
        :param logdet: log determinant

        :return: loss
        """
        # the original likelihood is sum over pixels, so we change to mean
        log_probs_q = (logdet + log_q - log(n_bins) * n_pixel) / (n_pixel * log(2))
        log_probs_q = log_probs_q.clamp(min=args.model_ll_clamp)

        loss = (
                log_probs_q -  # log likelihood model (Glow)
                log_p  # log likelihood data
        )

        return loss.mean()

    n_bins = 2.0 ** args.n_bits

    def gen_batch(batch_size=args.batch_size):
        z_sample = []
        z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
        for z in z_shapes:
            z_new = torch.randn(batch_size, *z) * args.temp
            z_sample.append(z_new.to(device))
        return z_sample

    pbar = trange(args.iters)

    global_iter = 0
    for i in pbar:
        batch = gen_batch()
        sampled_images = model.module.reverse(batch)
        # map to (-.5, .5)
        # todo: not sure what's the right transformation here
        # # map to (0, 255)
        # max_abs_val = .5
        # sampled_images = torch.clamp(sampled_images, -max_abs_val, max_abs_val)  # (-max_abs_val, max_abs_val)
        # sampled_images = (sampled_images + max_abs_val) / (2. * max_abs_val)  # (0, 1)
        # sampled_images = sampled_images * 255  # (0, 255)

        sampled_images = sampled_images / (sampled_images.abs().max() * 2.)  # approx. (-.5, .5)

        # pass through image gpt
        sampled_images_numpy = sampled_images.permute(0, 2, 3, 1).detach().cpu().numpy()
        # NOTE: expect channels last
        # clusters are in (-1, 1)
        # todo: maybe do this on Glow model's output? some other transformation?
        sampled_images_numpy = sampled_images_numpy * 2.  # (-1, 1)

        # sampled_images_numpy = (sampled_images_numpy / 255) - .5  # (-.5, .5)
        # sampled_images_numpy = sampled_images_numpy * 2.  # (-.1, .1)

        clustered_sampled_images = image_gpt.color_quantize(sampled_images_numpy)
        data_nll = image_gpt.eval_model(clustered_sampled_images)

        # # pass through Glow
        # # sampled_images = sampled_images.to(device)
        # if args.n_bits < 8:
        #     sampled_images = torch.floor(sampled_images / 2 ** (8 - args.n_bits))
        #
        # sampled_images = sampled_images / n_bins - 0.5
        # todo: why do we need that?
        if global_iter == 0:
            with torch.no_grad():
                log_p, logdet, _ = model.module(
                    sampled_images + torch.rand_like(sampled_images) / n_bins
                )
                global_iter += 1
                continue

        else:
            log_p, logdet, _ = model(sampled_images + torch.rand_like(sampled_images) / n_bins)

        # loss and metrics
        logdet = logdet.mean()
        data_log_likelihood = -torch.from_numpy(np.concatenate(data_nll)).to(device)
        loss = calc_loss(log_p=data_log_likelihood, log_q=log_p, logdet=logdet)
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
            f"Loss: {loss.item():.5f}; data LL: {-data_nll:.5f};  model LL: {model_ll.item():.5f}; "
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

                z_sample = gen_batch(batch_size=args.n_sample)
                try:
                    utils.save_image(
                        model_single.reverse(z_sample).cpu().data,
                        sample_path / f"{str(global_iter + 1).zfill(6)}.png",
                        normalize=True,
                        nrow=10,
                        # range=(-0.5, 0.5),
                        value_range=(-0.5, 0.5),
                        )
                except:
                    utils.save_image(
                        model_single.reverse(z_sample).cpu().data,
                        sample_path / f"{str(global_iter + 1).zfill(6)}.png",
                        normalize=True,
                        nrow=10,
                        range=(-0.5, 0.5),
                        # value_range=(-0.5, 0.5),
                        )
            if args.wandb:
                wandb.save((sample_path / f"{str(global_iter + 1).zfill(6)}.png").as_posix())

        global_iter += 1

        if i % args.model_every == 0:
            torch.save(
                model.state_dict(), model_path / f"model_{str(global_iter + 1).zfill(6)}.pt"
            )
            torch.save(
                optimizer.state_dict(), model_path / f"optim_{str(global_iter + 1).zfill(6)}.pt"
            )

    torch.save(
        model.state_dict(), model_path / f"model_end.pt"
    )
    torch.save(
        optimizer.state_dict(), model_path / f"optim_end.pt"
    )


if __name__ == "__main__":
    args = parser.parse_args()
    set_logger()
    # set seed
    set_seed(args.seed)

    device = torch.device(f"cuda:{args.pt_device[0]}" if torch.cuda.is_available() else "cpu")

    # Glow
    model_single = Glow(
        3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
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
        name = f"glow_imagegpt_reverse_kl_lr_{args.lr}_bs_{args.batch_size}_no_LU_{args.no_lu}_seed_{args.seed}"
        wandb.init(project="glow", entity='avivnav', name=name)
        wandb.config.update(args)

    train(args, model, optimizer, image_gpt)
