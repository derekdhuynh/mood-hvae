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
    def __init__(
            self, dset_path='/home/derek/mood_patches/brain_patches28_with_val.npz', train=True, transform=None, target_transform=None, val=False, toy=False, anom=False, ref=False, abdom=False, shuffle_dset=True):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.shuffle_dset = shuffle_dset

        self.dataset_path = dset_path

        if train:
            split = 'X_train'
        elif val and not train and not toy and not anom:
            split = 'X_val'
        elif not train and not val and not toy and not anom:
            split = 'X_test'
        elif anom:
            split = 'X_toy'

        print(self.dataset_path)
        print(split)
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
        dset_path='/home/derek/mood_patches/brain_patches28_with_val.npz',
        transform=None,
        target_transform=None,
        toy=False,
        anom=False,
        ref=False,
        abdom=False,
        shuffle=True
    ):
        super().__init__()

        transform = self.default_transform if transform is None else transform
        self.dataset = self._data_source(
            **self._split_args[split], dset_path=dset_path, transform=transform, toy=toy, anom=anom, ref=ref, abdom=abdom)

    @classmethod
    def get_argparser(cls):
        parser = argparse.ArgumentParser(description=cls.__name__)
        parser.add_argument(
            "--dset_path",
            default="/home/derek/mood_patches/brain_patches28_with_val.npz",
            help="Location of the dataset",
        )
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
            "--ref",
            type=str2bool,
            default=False,
            help="Datasets with 3 channels (images stacked ontop of each other",
        )
        parser.add_argument(
            "--abdom",
            type=str2bool,
            default=False,
            help="Abdominal dataset",
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

class BratsLinearNormalization(MOODPatchesRaw):
    default_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )
    def __init__(self,
            split=TRAIN_SPLIT,
            dset_path='/home/derek/mood_patches/brats_train_subvolume_patches_lin_norm.npz',
            transform=None,
            target_transform=None,
            toy=False,
            anom=False,
            ref=False,
            abdom=False,
            shuffle=True):
        transform = self.default_transform if split != TRAIN_SPLIT else self.default_transform 
        self.dataset = self._data_source(
            **self._split_args[split], dset_path=dset_path, transform=transform, toy=toy, anom=anom, ref=ref, abdom=abdom)


class BratsSubvolumeLinearNormalization(MOODPatchesRaw):
    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.RandomHorizontalFlip(0.5),
            # torchvision.transforms.RandomVerticalFlip(0.5),
            # torchvision.transforms.RandomApply([
            #     torchvision.transforms.RandomAffine(degrees=5, translate=(0, 0.05)),
            # ], p=0.7)
            #torchvision.transforms.RandomApply(
            #[
            #    torchvision.transforms.ColorJitter(
            #        brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25),
            #    torchvision.transforms.RandomRotation(30),
            #],
            #    p=0.3
            #)
        ]
    )

    default_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )

    def __init__(self,
            split=TRAIN_SPLIT,
            dset_path='/home/derek/mood_patches/brain_patches28_with_val.npz',
            transform=None,
            target_transform=None,
            toy=False,
            anom=False,
            ref=False,
            abdom=False,
            shuffle=True):

        if split == TRAIN_SPLIT:
            transform = self.train_transform
        else:
            transform = self.default_transform

        self.dataset = self._data_source(
            **self._split_args[split], dset_path=dset_path, transform=transform, toy=toy, anom=anom, ref=ref, abdom=abdom)

class MOODPatchesPctileKevin(MOODPatchesRaw):
    default_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            transforms.PctileKevin(0.95),
        ]
    )
