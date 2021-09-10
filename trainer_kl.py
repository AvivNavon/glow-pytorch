import logging
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Glow trainer", parents=[common_parser])
parser.add_argument("--batch-size", default=32, type=int, help="batch size")
parser.add_argument("--epochs", default=100, type=int, help="number of epochs")
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
parser.add_argument(
    '--loss', default='kl', type=str, choices=['kl', 'reverse-kl'], help="KL type for loss: ['kl', 'reverse-kl']"
)
parser.add_argument("--n_bits", default=5, type=int, help="number of bits")
parser.add_argument("--samples-every", default=500, type=int, help="samples every")
parser.add_argument("--model-every", default=50000, type=int, help="model every")
parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
parser.add_argument("--img_size", default=32, type=int, help="image size")
parser.add_argument("--workers", default=0, type=int, help="num workers")
parser.add_argument("--temp", default=0.7, type=float, help="temperature of sampling")
parser.add_argument("--n_sample", default=20, type=int, help="number of samples")
parser.add_argument("--sample-path", type=str, default='samples', help="Path to image directory")
parser.add_argument("--path", default='./data/imageGPT_Evaluation_Results_NLL.p', type=Path, help="Path to image directory")
parser.add_argument("--path-to-clusters", type=Path, help="Path to image directory", default='./data/kmeans_centers.npy')
parser.add_argument('--model-path', default='checkpoint', type=Path)
parser.add_argument("--seed", default=42, type=int, help="random seed")


def get_loader(path, clusters_path, sample_flag=False, device=None, batch_size=16):
    # todo: refactor
    # todo: we need to add augmentations like in train.py
    train, test = load_dataset_with_kl(path=path, clusters_path=clusters_path, sample_flag=sample_flag, device=device)
    dataset = TensorDataset(*train)

    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=args.workers)

    return loader


def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


# def calc_loss(log_p, logdet, image_size, n_bins):
#     # todo: use nn.KLDivLoss
#
#     # log_p = calc_log_p([z_list])
#     n_pixel = image_size * image_size * 3
#
#     loss = -log(n_bins) * n_pixel
#     loss = loss + logdet + log_p
#
#     return (
#         (-loss / (log(2) * n_pixel)).mean(),
#         (log_p / (log(2) * n_pixel)).mean(),
#         (logdet / (log(2) * n_pixel)).mean(),
#     )


def train(args, model, optimizer):
    sample_path = Path(args.sample_path)
    sample_path.mkdir(exist_ok=True, parents=True)

    model_path = args.model_path / f'{args.loss}'
    model_path.mkdir(exist_ok=True, parents=True)

    # loss
    reverse_kl = args.loss == 'reverse-kl'

    def calc_loss(p, q, reverse=False):
        """

        :param p: data likelihood
        :param q: model log-likelihood
        :param reverse: bool. Whether to use KL or reverse KL.

        :return: loss
        """
        if reverse:
            log_target = False
            # log_target = True
            # log_p = torch.log(p)
            loss = F.kl_div(q, p, log_target=log_target, reduction='batchmean')
        else:
            log_target = True
            log_p = torch.log(p)
            loss = F.kl_div(log_p, q, log_target=log_target, reduction='batchmean')
        return loss

    laoder = get_loader(args.path, args.path_to_clusters, device=device, batch_size=args.batch_size)
    n_bins = 2.0 ** args.n_bits

    z_sample = []
    z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
    for z in z_shapes:
        z_new = torch.randn(args.n_sample, *z) * args.temp
        z_sample.append(z_new.to(device))
    pbar = trange(args.epochs)

    global_iter = 0
    for epoch in pbar:
        for _, batch in enumerate(laoder):
            # todo: need to change once we have likelihood
            (image, likelihood) = (t.to(device) for t in batch)

            # image = image / 255.  # todo: we need to add transformations and augmentations

            if args.n_bits < 8:
                image = torch.floor(image / 2 ** (8 - args.n_bits))

            image = image / n_bins - 0.5

            if global_iter == 0:
                with torch.no_grad():
                    log_p, logdet, _ = model.module(
                        image + torch.rand_like(image) / n_bins
                    )
                    global_iter += 1
                    continue

            else:
                log_p, logdet, _ = model(image + torch.rand_like(image) / n_bins)

            # loss and metrics
            loss = calc_loss(p=likelihood.squeeze(), q=log_p, reverse=reverse_kl)
            logdet = logdet.mean()
            n_pixel = args.img_size * args.img_size * 3
            log_p = (log_p / (log(2) * n_pixel)).mean()
            log_det = (logdet / (log(2) * n_pixel)).mean()
            # loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
            model.zero_grad()
            loss.backward()
            # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
            warmup_lr = args.lr
            optimizer.param_groups[0]["lr"] = warmup_lr
            optimizer.step()

            pbar.set_description(
                f"Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}"
            )

            if args.wandb:
                wandb.log({
                    'loss': loss.item(),
                    'logP': log_p.item(),
                    'logdet': log_det.item()
                })

            if global_iter % args.samples_every == 0:
                with torch.no_grad():
                    utils.save_image(
                        model_single.reverse(z_sample).cpu().data,
                        sample_path / f"{str(global_iter + 1).zfill(6)}.png",
                        normalize=True,
                        nrow=10,
                        # range=(-0.5, 0.5),
                        value_range=(-0.5, 0.5),
                        )
                if args.wandb:
                    wandb.save((sample_path / f"{str(global_iter + 1).zfill(6)}.png").as_posix())

            global_iter += 1

        if epoch % args.model_every == 0:
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

    model_single = Glow(
        3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
    )
    model = nn.DataParallel(model_single)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # set seed
    set_seed(args.seed)
    # wandb
    if args.wandb:
        name = f"glow_imagegpt_KL_lr_{args.lr}_bs_{args.batch_size}_no_LU_{args.no_lu}_seed_{args.seed}"
        wandb.init(project="glow", entity='avivnav', name=name)
        wandb.config.update(args)

    train(args, model, optimizer)