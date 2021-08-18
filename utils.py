import argparse
import logging
import random
import numpy as np
import torch


common_parser = argparse.ArgumentParser(add_help=False)
common_parser.add_argument('--wandb', dest='wandb', action='store_true')
common_parser.add_argument('--no-wandb', dest='wandb', action='store_false')
common_parser.set_defaults(wandb=True)


def set_logger():
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
