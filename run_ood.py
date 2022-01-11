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

parser = argparse.ArgumentParser(description="Glow trainer", parents=[common_parser])
parser.add_argument("--batch-size", default=256, type=int, help="batch size")
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
parser.add_argument("--ood-data-name", default='cifar10', choices=['mnist', 'cifar10'], type=str)
parser.add_argument("--ood-data-path", default='tmp/', type=str)
parser.add_argument("--n_bits", default=5, type=int, help="number of bits")
parser.add_argument("--img_size", default=32, type=int, help="image size")
parser.add_argument("--workers", default=0, type=int, help="num workers")
parser.add_argument('--model-path-nll', default='checkpoint/kl/model_050001.pt', type=Path)
parser.add_argument('--model-path-reverse', default='checkpoint/kl/model_050001.pt', type=Path)
parser.add_argument("--path", default='data', type=Path, help="Path to image directory")
parser.add_argument("--path-to-clusters", type=Path, help="Path to image directory", default='data')
parser.add_argument("--out-path", default='ood-results', type=Path, help="Path to outputs")
parser.add_argument("--seed", default=42, type=int, help="random seed")


args = parser.parse_args()
set_logger()

# load model - NLL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(path, device=device):
    model = Glow(
        3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
    )
    model = nn.DataParallel(model)
    model = model.to(device)
    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    return model


model_nll = load_model(args.model_path_nll)
model_reverse = load_model(args.model_path_reverse)


# in distribution data
def get_loader_in_dist(path, clusters_path, sample_flag=False, device=None, batch_size=16):
    train, test = load_datasets(path=path, clusters_path=clusters_path, sample_flag=sample_flag, device=device)
    # NOTE: taking test here!
    dataset = TensorDataset(test / 255.)  # to (0, 1)

    loader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=args.workers)

    return loader


in_dist_loader = get_loader_in_dist(args.path, args.path_to_clusters, device=device, batch_size=args.batch_size)

# out dist data
dataset_class = dict(mnist=MNIST, cifar10=CIFAR10)[args.ood_data_name]
image_transform = [
    # transforms.ToTensor(),
]
if args.ood_data_name == 'mnist':
    image_transform.append(transforms.Pad(padding=2))
    image_transform.append(transforms.ToTensor())
    image_transform.append(lambda image: image.repeat(3, 1, 1))  # duplicate along channel dim.

else:
    image_transform.append(transforms.ToTensor())

transform = transforms.Compose(image_transform)
ood_dataset = dataset_class(args.ood_data_path, train=False, transform=transform, download=True)
out_dist_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=args.batch_size, shuffle=False)


# evaluate model
@torch.no_grad()
def eval_model(model, loader, desc=None):
    model.eval()
    log_likelihood = []
    for batch in tqdm(loader, desc=desc, total=len(loader)):
        batch = list((t.to(device) for t in batch))
        if len(batch) == 2:
            batch = batch[:1]
        image = batch[0]

        n_bins = 2.0 ** args.n_bits
        n_pixel = args.img_size * args.img_size * 3

        # assume image in (0, 1)
        image = image * 255.

        if args.n_bits < 8:
            image = torch.floor(image / 2 ** (8 - args.n_bits))

        image = image / n_bins - 0.5

        # log_p, logdet, _ = model(image + torch.rand_like(image) / n_bins)  # todo: remove random?
        log_p, logdet, _ = model(image)
        curr_log_likelihood = (logdet + log_p - log(n_bins) * n_pixel) / (n_pixel * log(2))
        log_likelihood.extend(curr_log_likelihood.cpu().detach().numpy().tolist())
    return log_likelihood


# compute log likelihood
out_path = args.out_path
out_path.mkdir(exist_ok=True, parents=True)

# NLL model

# NOTE: at the end we used JS instead of R-KL, hence the name changing

results = dict()
results['nll_in'] = eval_model(model_nll, in_dist_loader, desc='NLL in')
results['nll_out'] = eval_model(model_nll, out_dist_loader, desc='NLL out')
results['js_in'] = eval_model(model_reverse, in_dist_loader, desc='JS in')
results['js_out'] = eval_model(model_reverse, out_dist_loader, desc='JS out')

with open((out_path / f"{args.ood_data_name}.json").as_posix(), "w") as f:
    json.dump(results, f)
