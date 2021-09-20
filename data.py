import os
from pathlib import Path
import logging

import pickle
import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split


def load_h5_dataset(path: Path):
    data = []
    flagOneFile = 0
    for filename in os.listdir(path):
        if flagOneFile:
            break
        if filename.endswith(".h5"):
            with h5py.File(path / filename, "r") as f:
                a_group_key = list(f.keys())[0]
                # Get the data
                temp = list(f[a_group_key])
                data.append(temp[1:])
                flagOneFile = 0
            continue
        else:
            continue
    data_flat = [item for sublist in data for item in sublist]
    data_flat = np.stack(data_flat, axis=0)
    precent_train_test_split = 0.7
    train = data_flat[:int(np.floor(precent_train_test_split * data_flat.shape[0])), :]
    test = data_flat[int(np.floor(precent_train_test_split * data_flat.shape[0])) + 1:, :]

    if not os.path.isfile(path / 'test_imagegpt.h5'):
        print("Saving H5DF files...")
        test_h5 = h5py.File(path / 'test_imagegpt.h5', 'w')
        test_h5.create_dataset('test', data=test)
        train_h5 = h5py.File(path / 'train_imagegpt.h5', 'w')
        train_h5.create_dataset('train', data=train)

    return train, test


class ImageMapping:
    def __init__(self, cluster_path, sample_flag=False, device=None):
        self.clusters = torch.from_numpy(np.load(cluster_path)).float().to(device)
        self.sample_flag = sample_flag
        self.device = device

    def map_image(self, x):
        x = torch.from_numpy(x)  # .to(self.device)
        x = torch.round(127.5 * (self.clusters[x.long()] + 1.0))

        # Training --> x.shape = torch.Size([128, 1024, 3])
        # Sampling --> x.shape = torch.Size([1024, 1, 3])

        if self.sample_flag:
            x = x.permute(1, 0, 2)

        # Sampling --> x.shape = torch.Size([1, 1024, 3])

        x = x[:,:,None,:]

        # Training --> x.shape = torch.Size([128, 1024, 1, 3])
        # Sampling --> x.shape = torch.Size([1, 1024, 1, 3])

        x = torch.reshape(x, [x.shape[0], 32, 32,x.shape[3]])

        # Training --> x.shape = torch.Size([128, 32, 32, 3])
        # Sampling --> x.shape = torch.Size([1, 32, 32, 3])

        x = x.permute(0, 3, 1, 2)

        # Training --> x.shape = torch.Size([128, 3, 32, 32])
        # Sampling --> x.shape =torch.Size([1, 3, 32, 32])

        return x


def load_datasets(path: Path, clusters_path, sample_flag=False, device=None):
    train, test = load_h5_dataset(Path(path))
    mapping = ImageMapping(clusters_path, sample_flag=sample_flag, device=device)
    train = mapping.map_image(train)
    test = mapping.map_image(test)
    return train, test


def load_dataset_with_kl(path, clusters_path, sample_flag=False, device=None, test_size=.2, seed=42):
    model_table = pickle.load(open(path, "rb"))
    train, test = train_test_split(model_table, test_size=test_size, random_state=seed)
    train, train_likelihood = np.vstack([t[0] for t in train]), np.vstack([t[1] for t in train])
    test, test_likelihood = np.vstack([t[0] for t in test]), np.vstack([t[1] for t in test])
    mapping = ImageMapping(clusters_path, sample_flag=sample_flag, device=device)
    train = mapping.map_image(train)
    test = mapping.map_image(test)
    return (
        (train, torch.from_numpy(train_likelihood.astype(np.float32))),
        (test, torch.from_numpy(test_likelihood.astype(np.float32)))
    )


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    # path = './data'
    # train, test = load_h5_dataset(Path(path))
    # logging.info(f"train shape: {train.shape}, test shape: {test.shape}")
    #
    # mapping = ImageMapping('./data/kmeans_centers.npy')
    # x = mapping.map_image(test)
    # logging.info(x.shape)
    # logging.info(x[0])

    train, test = load_dataset_with_kl('./data/imageGPT_Evaluation_Results_NLL.p', './data/kmeans_centers.npy')
    print('debug')
