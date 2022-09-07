import argparse
import logging
import os
import PIL
import tarfile

import tqdm
import numpy as np
import torch
import torch.utils.data as data
import torchvision


from urllib.request import urlretrieve

import oodd
from oodd.utils.argparsing import str2bool
from oodd.datasets import transforms
from oodd.datasets import BaseDataset
from oodd.constants import TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, TOY_SPLIT, DATA_DIRECTORY

def load_dataset_from_file(path):
    return np.load(path)

class MOODPatches(data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, val=False, toy=False, anom=False, big=False, ref=False, shuffle_dset=True):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        #self.dataset_path = '/home/derek/mood_patches/brain_patches28_with_val_black.npz'
        self.dataset_path = '/home/derek/mood_patches/brain_patches28_with_val.npz'
        self.shuffle_dset = shuffle_dset

        if train:
            split = 'X_train'
            if big:
                self.dataset_path = '/home/derek/mood_patches/120k_raw_brain_patches.npz'
            elif ref:
                self.dataset_path = '/home/derek/mood_patches/ref_brain_patches_28.npz'
        elif val and not train and not toy and not anom:
            split = 'X_val'
            if ref:
                self.dataset_path = '/home/derek/mood_patches/ref_brain_patches_28.npz'
        elif not train and not val and not toy and not anom:
            split = 'X_test'
            if ref:
                self.dataset_path = '/home/derek/mood_patches/ref_brain_patches_28.npz'
        elif anom:
            split = 'X_toy'
            if ref:
                self.dataset_path = '/home/derek/mood_patches/ref_toy_brain_patches_28.npz'
            else:
                self.dataset_path = '/home/derek/mood_patches/brain_toy_patches_only_anom.npz'
        elif toy:
            split = 'X_toy'
            self.dataset_path = '/home/derek/mood_patches/brain_toy_patches28_with_labels.npz'

        dset = load_dataset_from_file(self.dataset_path)
        self.examples = dset[split]

        if toy or anom:
            self.targets = dset['y_toy']
        else:
            self.targets = torch.arange(self.examples.shape[0])

        self.shuffle(seed=19690720)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (image, target) where target is idx of the target class.
        """
        example, target = self.examples[idx], self.targets[idx]
        example = self.examples[idx].astype('float32')

        #example = PIL.Image.fromarray(example.squeeze())  # 28x28 to PIL image

        if self.transform is not None:
            example = self.transform(example)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return example, target

    def shuffle(self, seed):
        if self.shuffle_dset:
            rng = np.random.default_rng(seed=seed)
            rand_idx = rng.permutation(list(range(len(self.examples))))
            self.examples = self.examples[rand_idx]
            self.targets = self.targets[rand_idx]

    def __repr__(self):
        dataset_path, train, transform = (
            self.dataset_path,
            self.train,
            self.transform,
            self.target_transform,
        )
        fmt_str = f"MOOD_Patches({dataset_path=}, {train=}, {transform=})"
        return fmt_str

    def __len__(self):
        return len(self.examples)

class MOODPatchesRaw(BaseDataset):
    _data_source = MOODPatches
    _split_args = {TRAIN_SPLIT: {"train": True, "val": False}, VAL_SPLIT: {"train": False, "val": True}, TEST_SPLIT: {"train": False, "val": False}}

    default_transform = torchvision.transforms.ToTensor()

    def __init__(
        self,
        split=TRAIN_SPLIT,
        root=DATA_DIRECTORY,
        transform=None,
        target_transform=None,
        toy=False,
        anom=False,
        big=False,
        ref=False,
        shuffle=True
    ):
        super().__init__()

        transform = self.default_transform if transform is None else transform
        self.dataset = self._data_source(
            **self._split_args[split],transform=transform, toy=toy, anom=anom, big=big, ref=ref)

    @classmethod
    def get_argparser(cls):
        parser = argparse.ArgumentParser(description=cls.__name__)
        parser.add_argument(
            "--toy",
            type=str2bool,
            default=False,
            help="If true, will use the patches sampled from the toy brains",
        )
        parser.add_argument(
            "--shuffle",
            type=str2bool,
            default=True,
            help="Decide whether or not to shuffle the split",
        )
        parser.add_argument(
            "--anom",
            type=str2bool,
            default=False,
            help="Only use dataset with 10% anomalous voxels",
        )
        parser.add_argument(
            "--big",
            type=str2bool,
            default=False,
            help="Larger dataset containing 120k patches",
        )
        parser.add_argument(
            "--ref",
            type=str2bool,
            default=False,
            help="Datasets with 3 channels (images stacked ontop of each other",
        )
        return parser

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class MOODPatchesPctileChauhan(MOODPatchesRaw):
    default_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            transforms.PctileChauhan(0.05),
        ]
    )
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class MOODPatchesPctileKevin(MOODPatchesRaw):
    default_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            transforms.PctileKevin(0.95),
        ]
    )
