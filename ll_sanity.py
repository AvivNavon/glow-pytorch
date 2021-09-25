import argparse
import json
import logging
from pathlib import Path
import json

import numpy as np
from math import log
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm, trange
import wandb

from model import Glow
from utils import common_parser, set_seed, set_logger
from data import load_datasets
from torchvision.datasets import CIFAR10, MNIST
from imagegpt.imagegpt import ImageGPT


parser = argparse.ArgumentParser(description="Glow trainer", parents=[common_parser])
parser.add_argument("--batch-size", default=2, type=int, help="batch size")
parser.add_argument("--n-batch", default=250, type=int, help="number of batch")
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
parser.add_argument("--temp", default=0.7, type=float, help="temperature of sampling")
parser.add_argument('--imagegpt-artifact', default='../image-gpt/artifacts', type=Path)
parser.add_argument("--n_bits", default=5, type=int, help="number of bits")
parser.add_argument("--img_size", default=32, type=int, help="image size")
parser.add_argument("--workers", default=0, type=int, help="num workers")
parser.add_argument('--glow-path', default='checkpoint/kl/model_050001.pt', type=Path)
parser.add_argument("--tf-device", nargs="+", type=int, default=[0], help="GPU devices for tf")
parser.add_argument("--pt-device", nargs="+", type=int, default=[0], help="GPU devices for pt")
# parser.add_argument("--path", default='data', type=Path, help="Path to image directory")
# parser.add_argument("--path-to-clusters", type=Path, help="Path to image directory", default='data')
parser.add_argument("--out-path", default='ood-results', type=Path, help="Path to outputs")
parser.add_argument("--seed", default=42, type=int, help="random seed")


args = parser.parse_args()
set_logger()

# load model - NLL
device = torch.device(f"cuda:{args.pt_device[0]}" if torch.cuda.is_available() else "cpu")


# glow
def load_glow(path):
    model = Glow(
        3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu, reverse_log_det=True
    )
    model = nn.DataParallel(model, device_ids=args.pt_device)
    model = model.to(device)
    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    return model


glow = load_glow(args.glow_path)


# image gpt
artifacts_path = args.imagegpt_artifact
image_gpt = ImageGPT(
    batch_size=args.batch_size,
    devices=args.tf_device,
    ckpt_path=(artifacts_path / "model.ckpt-1000000").as_posix(),
    color_cluster_path=(artifacts_path / "kmeans_centers.npy").as_posix(),
)


# # in distribution data
# def get_loader_in_dist(path, clusters_path, sample_flag=False, device=None, batch_size=16):
#     train, test = load_datasets(path=path, clusters_path=clusters_path, sample_flag=sample_flag, device=device)
#     # NOTE: taking test here!
#     dataset = TensorDataset(test / 255.)  # to (0, 1)
#
#     loader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=args.workers)
#
#     return loader

def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


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


n_bins = 2.0 ** args.n_bits
n_pixel = args.img_size * args.img_size * 3
ratio = []
total = 0.
with torch.no_grad():
    with tqdm(range(args.n_batch), total=args.n_batch) as iterator:
        for i in iterator:
            z_batch, log_probs = gen_batch()
            sampled_images, logdet = glow.module.reverse(z_batch)
            log_pz = sum(log_probs)
            log_pz = log_pz.to(device)

            # todo: we maybe need logdet for eac image and not averaged over bacth
            glow_log_likelihood = (logdet + log_pz - log(n_bins) * n_pixel) / (n_pixel * log(2))
            glow_likelihood = torch.exp(glow_log_likelihood).cpu().numpy()

            # pass through image gpt
            sampled_images = torch.clamp(sampled_images, -.5, .5)
            sampled_images_numpy = sampled_images.permute(0, 2, 3, 1).detach().cpu().numpy()
            # NOTE: expect channels last
            # clusters are in (-1, 1)
            sampled_images_numpy = sampled_images_numpy * 2.
            clustered_sampled_images = image_gpt.color_quantize(sampled_images_numpy)
            data_nll = image_gpt.eval_model(clustered_sampled_images)
            image_gpt_likelihood = np.exp(-np.concatenate(data_nll))

            ratio.extend((image_gpt_likelihood / glow_likelihood).tolist())
            total += len(glow_likelihood)
            iterator.set_description(f"iter {i+1}: ratio mean: {np.mean(ratio):.3f} (number of samples {total:d})")
