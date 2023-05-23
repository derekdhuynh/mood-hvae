"""Module with datasets from torchvision"""

import argparse
import logging
import os

import torchvision

import oodd.constants

from oodd.datasets import transforms
from oodd.constants import TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, DATA_DIRECTORY
from oodd.utils.argparsing import str2bool

from .dataset_base import BaseDataset


LOGGER = logging.getLogger(name=__file__)


TRANSFORM_DEQUANTIZE_8BIT = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        transforms.Scale(a=0, b=255, min_val=0, max_val=1),  # Scale to [0, 1]
        transforms.Dequantize(),  # Add U(0, 1) noise, becomes [0, 256]
        transforms.Scale(a=0, b=1, min_val=0, max_val=256),  # Scale to [0, 1]
    ]
)


TRANSFORM_BINARIZE = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        transforms.Binarize(resample=True),
    ]
)


def memoize(func):
    cache = dict()

    def memoized_func(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result

    return memoized_func


class TorchVisionDataset(BaseDataset):
    _data_source = lambda x: x
    _split_args = dict()
    default_transform = lambda x: x
    root_subdir = ""

    def __init__(
        self,
        split=oodd.constants.TRAIN_SPLIT,
        root=DATA_DIRECTORY,
        transform=None,
        target_transform=None,
        dynamic: bool = False,
    ):
        super().__init__()

        self.split = split
        self.root = root if not self.root_subdir else os.path.join(root, self.root_subdir)
        self.transform = self.default_transform if transform is None else transform
        self.target_transform = target_transform
        self.dynamic = dynamic
        if self.dynamic:
            LOGGER.info("Running with caching")
            self.item_getter = memoize(self.item_getter)

        if self.root_subdir == 'LSUN' or self.root_subdir =='FER2013':
            self.dataset = self._data_source(
                **self._split_args[split],
                root=self.root,
                transform=self.transform,
                target_transform=target_transform,
            )

        elif self.root_subdir == 'Places365':
            self.dataset = self._data_source(
                **self._split_args[split],
                root=self.root,
                small=True,
                transform=self.transform,
                target_transform=target_transform,
                download=False
            )

        elif self.root_subdir == 'FakeData':
            self.dataset = self._data_source(
                size=10000,
                image_size=(3, 32, 32),
                num_classes=10,
                transform=self.transform,
                target_transform=target_transform,
            )

        elif self.root_subdir == "SUN397" or self.root_subdir =="EuroSAT":
            self.dataset = self._data_source(
                root=self.root,
                transform=self.transform,
                target_transform=target_transform,
                download=True
            )

        else:
            self.dataset = self._data_source(
                **self._split_args[split],
                root=self.root,
                transform=self.transform,
                target_transform=target_transform,
                download=True
            )

    @classmethod
    def get_argparser(cls):
        parser = argparse.ArgumentParser(description=cls.__name__)
        parser.add_argument("--root", type=str, default=DATA_DIRECTORY, help="Data storage location")
        parser.add_argument(
            "--dynamic",
            type=str2bool,
            default=False,
            help="If False, all values are cached in the first epoch to disable dynamic resampling",
        )
        return parser

    def item_getter(self, idx):
        return self.dataset[idx]

    def __getitem__(self, idx):
        return self.item_getter(idx)

    def __len__(self):
        return len(self.dataset)


class MNISTQuantized(TorchVisionDataset):
    """MNIST dataset serving quantized pixel values in [0, 1] (256 unique values)"""

    _data_source = torchvision.datasets.MNIST
    _split_args = {TRAIN_SPLIT: {"train": True}, VAL_SPLIT: {"train": False}}
    default_transform = torchvision.transforms.ToTensor()


class MNISTBinarized(MNISTQuantized):
    """MNIST dataset serving quantized pixel values in {0, 1} (2 unique values)"""

    default_transform = TRANSFORM_BINARIZE


class MNISTDequantized(MNISTQuantized):
    """MNIST dataset serving dequantized pixel values in [0, 1] via 'x <- (x + u) / (255 + 1))'"""

    default_transform = TRANSFORM_DEQUANTIZE_8BIT


class FashionMNISTQuantized(TorchVisionDataset):
    """FashionMNIST dataset serving quantized pixel values in [0, 1] (256 unique values)"""

    _data_source = torchvision.datasets.FashionMNIST
    _split_args = {TRAIN_SPLIT: {"train": True}, VAL_SPLIT: {"train": False}}
    default_transform = torchvision.transforms.ToTensor()


class FashionMNISTBinarized(FashionMNISTQuantized):
    """FashionMNIST dataset serving quantized pixel values in {0, 1} (2 unique values)"""

    default_transform = TRANSFORM_BINARIZE


class FashionMNISTDequantized(FashionMNISTQuantized):
    """FashionMNIST dataset serving dequantized pixel values in [0, 1] via 'x <- (x + u) / (255 + 1))'"""

    default_transform = TRANSFORM_DEQUANTIZE_8BIT


class KMNISTQuantized(TorchVisionDataset):
    """KMNIST dataset serving quantized pixel values in [0, 1] (256 unique values)"""

    _data_source = torchvision.datasets.KMNIST
    _split_args = {TRAIN_SPLIT: {"train": True}, VAL_SPLIT: {"train": False}}
    default_transform = torchvision.transforms.ToTensor()


class KMNISTBinarized(KMNISTQuantized):
    """KMNIST dataset serving quantized pixel values in {0, 1} (2 unique values)"""

    default_transform = TRANSFORM_BINARIZE


class KMNISTDequantized(KMNISTQuantized):
    """KMNIST dataset serving dequantized pixel values in [0, 1] via 'x <- (x + u) / (255 + 1))'"""

    default_transform = TRANSFORM_DEQUANTIZE_8BIT


class CIFAR10Quantized(TorchVisionDataset):
    _data_source = torchvision.datasets.CIFAR10  # Shape [N, 32, 32, 3]
    _split_args = {TRAIN_SPLIT: {"train": True}, VAL_SPLIT: {"train": False}, TEST_SPLIT: {"train": False}}
    default_transform = torchvision.transforms.ToTensor()
    root_subdir = "CIFAR10"


class CIFAR10Dequantized(CIFAR10Quantized):
    default_transform = TRANSFORM_DEQUANTIZE_8BIT


class CIFAR10DequantizedGrey(CIFAR10Quantized):
    default_transform = torchvision.transforms.Compose(
        [TRANSFORM_DEQUANTIZE_8BIT, transforms.Grayscale(num_output_channels=3)]
    )


class SVHNQuantized(TorchVisionDataset):
    _data_source = torchvision.datasets.SVHN  # Shape [N, 32, 32, 3]
    _split_args = {TRAIN_SPLIT: {"split": "train"}, VAL_SPLIT: {"split": "extra"}, TEST_SPLIT: {"split": "test"}}
    default_transform = torchvision.transforms.ToTensor()
    root_subdir = "SVHN"


class SVHNDequantized(SVHNQuantized):
    default_transform = TRANSFORM_DEQUANTIZE_8BIT


class SVHNDequantizedGrey(SVHNQuantized):
    default_transform = torchvision.transforms.Compose(
        [TRANSFORM_DEQUANTIZE_8BIT, transforms.Grayscale(num_output_channels=3)]
    )




class LSUNQuantized(TorchVisionDataset):
    _data_source = torchvision.datasets.LSUN  # Shape [N, 32, 32, 3]
    # _split_args = {TRAIN_SPLIT: {"split": "train"}, VAL_SPLIT: {"split": "extra"}, TEST_SPLIT: {"split": "test"}}
    _split_args = {TRAIN_SPLIT: {"classes": 'train'}, VAL_SPLIT: {"classes": 'val'}, TEST_SPLIT: {"classes": 'test'}}
    default_transform = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(size=32), torchvision.transforms.ToTensor()])  # Shape [N, 32, 32, 3]
    root_subdir = "LSUN"


class LSUNDequantized(LSUNQuantized):
    default_transform = TRANSFORM_DEQUANTIZE_8BIT


class LSUNDequantizedGrey(LSUNQuantized):
    default_transform = torchvision.transforms.Compose(
        [TRANSFORM_DEQUANTIZE_8BIT, transforms.Grayscale(num_output_channels=3)]
    )





class CIFAR100Quantized(TorchVisionDataset):
    _data_source = torchvision.datasets.CIFAR100  # Shape [N, 32, 32, 3]
    _split_args = {TRAIN_SPLIT: {"train": True}, VAL_SPLIT: {"train": False}, TEST_SPLIT: {"train": False}}
    default_transform = torchvision.transforms.ToTensor()
    root_subdir = "CIFAR100"


class CIFAR100Dequantized(CIFAR100Quantized):
    default_transform = TRANSFORM_DEQUANTIZE_8BIT


class CIFAR100DequantizedGrey(CIFAR100Quantized):
    default_transform = torchvision.transforms.Compose(
        [TRANSFORM_DEQUANTIZE_8BIT, transforms.Grayscale(num_output_channels=3)]
    )



class STL10Quantized(TorchVisionDataset):
    _data_source = torchvision.datasets.STL10  # Shape [N, 32, 32, 3]
    _split_args =  {TRAIN_SPLIT: {"split": 'train'}, VAL_SPLIT: {"split": 'valid'}, TEST_SPLIT: {"split": 'test'}}
    default_transform = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(size=32), torchvision.transforms.ToTensor()])  # Shape [N, 32, 32, 3]
    root_subdir = "STL10"


class STL10Dequantized(STL10Quantized):
    default_transform = TRANSFORM_DEQUANTIZE_8BIT


class STL10DequantizedGrey(STL10Quantized):
    default_transform = torchvision.transforms.Compose(
        [TRANSFORM_DEQUANTIZE_8BIT, transforms.Grayscale(num_output_channels=3)]
    )



class CelebAQuantized(TorchVisionDataset):
    _data_source = torchvision.datasets.CelebA
    _split_args = {TRAIN_SPLIT: {"split": 'train'}, VAL_SPLIT: {"split": 'valid'}, TEST_SPLIT: {"split": 'test'}}
    # default_transform = torchvision.transforms.ToTensor()
    root_subdir = "CelebA"
    default_transform = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(size=32), torchvision.transforms.ToTensor()])  # Shape [N, 32, 32, 3]
    # default_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(size=32), torchvision.transforms.ToTensor()])  # Shape [N, 32, 32, 3]


class CelebADequantized(CelebAQuantized):
    default_transform = TRANSFORM_DEQUANTIZE_8BIT


class CelebADequantizedGrey(CelebAQuantized):
    default_transform = torchvision.transforms.Compose(
        [TRANSFORM_DEQUANTIZE_8BIT, transforms.Grayscale(num_output_channels=3)]
    )


class Food101Quantized(TorchVisionDataset):
    _data_source = torchvision.datasets.Food101
    _split_args = {TRAIN_SPLIT: {"split": 'train'}, VAL_SPLIT: {"split": 'test'}, TEST_SPLIT: {"split": 'test'}}
    # default_transform = torchvision.transforms.ToTensor()
    root_subdir = "Food101"
    default_transform = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(size=32), torchvision.transforms.ToTensor()])  # Shape [N, 32, 32, 3]


class Food101Dequantized(Food101Quantized):
    default_transform = TRANSFORM_DEQUANTIZE_8BIT


class Food101DequantizedGrey(Food101Quantized):
    default_transform = torchvision.transforms.Compose(
        [TRANSFORM_DEQUANTIZE_8BIT, transforms.Grayscale(num_output_channels=3)]
    )


class Flowers102Quantized(TorchVisionDataset):
    _data_source = torchvision.datasets.Flowers102
    _split_args = {TRAIN_SPLIT: {"split": 'train'}, VAL_SPLIT: {"split": 'test'}, TEST_SPLIT: {"split": 'test'}}
    # default_transform = torchvision.transforms.ToTensor()
    root_subdir = "Flowers102"
    default_transform = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(size=32), torchvision.transforms.ToTensor()])  # Shape [N, 32, 32, 3]
    # default_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(size=32), torchvision.transforms.ToTensor()])  # Shape [N, 32, 32, 3]


class Flowers102Dequantized(Flowers102Quantized):
    default_transform = TRANSFORM_DEQUANTIZE_8BIT


class Flowers102DequantizedGrey(Flowers102Quantized):
    default_transform = torchvision.transforms.Compose(
        [TRANSFORM_DEQUANTIZE_8BIT, transforms.Grayscale(num_output_channels=3)]
    )


class Places365Quantized(TorchVisionDataset):
    _data_source = torchvision.datasets.Places365
    _split_args = {TRAIN_SPLIT: {"split": 'train-standard'}, VAL_SPLIT: {"split": 'train-standard'}, TEST_SPLIT: {"split": 'val'}}
    # default_transform = torchvision.transforms.ToTensor()
    root_subdir = "Places365"
    default_transform = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(size=32), torchvision.transforms.ToTensor()])  # Shape [N, 32, 32, 3]
    # default_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(size=32), torchvision.transforms.ToTensor()])  # Shape [N, 32, 32, 3]


class Places365Dequantized(Places365Quantized):
    default_transform = TRANSFORM_DEQUANTIZE_8BIT


class Places365DequantizedGrey(Places365Quantized):
    default_transform = torchvision.transforms.Compose(
        [TRANSFORM_DEQUANTIZE_8BIT, transforms.Grayscale(num_output_channels=3)]
    )


class FakeDataQuantized(TorchVisionDataset):
    _data_source = torchvision.datasets.FakeData
    _split_args = {TRAIN_SPLIT: {"size": 10000}, VAL_SPLIT: {"size": 10000}, TEST_SPLIT: {"size": 10000}}
    # default_transform = torchvision.transforms.ToTensor()
    root_subdir = "FakeData"
    default_transform = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(size=32), torchvision.transforms.ToTensor()])  # Shape [N, 32, 32, 3]


class FakeDataDequantized(FakeDataQuantized):
    default_transform = TRANSFORM_DEQUANTIZE_8BIT


class FakeDataDequantizedGrey(FakeDataQuantized):
    default_transform = torchvision.transforms.Compose(
        [TRANSFORM_DEQUANTIZE_8BIT, transforms.Grayscale(num_output_channels=3)]
    )


class LFWPeopleQuantized(TorchVisionDataset):
    _data_source = torchvision.datasets.LFWPeople
    _split_args = {TRAIN_SPLIT: {"split": 'train'}, VAL_SPLIT: {"split": 'test'}, TEST_SPLIT: {"split": 'test'}}
    # default_transform = torchvision.transforms.ToTensor()
    root_subdir = "LFWPeople"
    default_transform = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(size=32), torchvision.transforms.ToTensor()])  # Shape [N, 32, 32, 3]
    # default_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(size=32), torchvision.transforms.ToTensor()])  # Shape [N, 32, 32, 3]


class LFWPeopleDequantized(LFWPeopleQuantized):
    default_transform = TRANSFORM_DEQUANTIZE_8BIT


class LFWPeopleDequantizedGrey(LFWPeopleQuantized):
    default_transform = torchvision.transforms.Compose(
        [TRANSFORM_DEQUANTIZE_8BIT, transforms.Grayscale(num_output_channels=3)]
    )


class SUN397Quantized(TorchVisionDataset):
    _data_source = torchvision.datasets.SUN397
    _split_args = {TRAIN_SPLIT: {"split": 'train'}, VAL_SPLIT: {"split": 'test'}, TEST_SPLIT: {"split": 'test'}}
    # default_transform = torchvision.transforms.ToTensor()
    root_subdir = "SUN397"
    default_transform = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(size=32), torchvision.transforms.ToTensor()])  # Shape [N, 32, 32, 3]


class SUN397PeopleDequantized(SUN397Quantized):
    default_transform = TRANSFORM_DEQUANTIZE_8BIT


class SUN397DequantizedGrey(SUN397Quantized):
    default_transform = torchvision.transforms.Compose(
        [TRANSFORM_DEQUANTIZE_8BIT, transforms.Grayscale(num_output_channels=3)]
    )


class GTSRBQuantized(TorchVisionDataset):
    _data_source = torchvision.datasets.GTSRB
    _split_args = {TRAIN_SPLIT: {"split": 'train'}, VAL_SPLIT: {"split": 'test'}, TEST_SPLIT: {"split": 'test'}}
    # default_transform = torchvision.transforms.ToTensor()
    root_subdir = "GTSRB"
    default_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(size=[32,32]), torchvision.transforms.ToTensor()])  # Shape [N, 32, 32, 3]


class GTSRBDequantized(GTSRBQuantized):
    default_transform = TRANSFORM_DEQUANTIZE_8BIT


class GTSRBDequantizedGrey(GTSRBQuantized):
    default_transform = torchvision.transforms.Compose(
        [TRANSFORM_DEQUANTIZE_8BIT, transforms.Grayscale(num_output_channels=3)]
    )


class PCAMQuantized(TorchVisionDataset):
    _data_source = torchvision.datasets.PCAM
    _split_args = {TRAIN_SPLIT: {"split": 'train'}, VAL_SPLIT: {"split": 'val'}, TEST_SPLIT: {"split": 'test'}}
    # default_transform = torchvision.transforms.ToTensor()
    root_subdir = "PCAM"
    default_transform = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(size=32), torchvision.transforms.ToTensor()])  # Shape [N, 32, 32, 3]


class PCAMDequantized(PCAMQuantized):
    default_transform = TRANSFORM_DEQUANTIZE_8BIT


class PCAMDequantizedGrey(PCAMQuantized):
    default_transform = torchvision.transforms.Compose(
        [TRANSFORM_DEQUANTIZE_8BIT, transforms.Grayscale(num_output_channels=3)]
    )


class FER2013Quantized(TorchVisionDataset):
    _data_source = torchvision.datasets.FER2013
    _split_args = {TRAIN_SPLIT: {"split": 'train'}, VAL_SPLIT: {"split": 'val'}, TEST_SPLIT: {"split": 'test'}}
    # default_transform = torchvision.transforms.ToTensor()
    root_subdir = "FER2013"
    default_transform = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(size=32), torchvision.transforms.ToTensor()])  # Shape [N, 32, 32, 3]


class FER2013Dequantized(FER2013Quantized):
    default_transform = TRANSFORM_DEQUANTIZE_8BIT


class FER2013DequantizedGrey(FER2013Quantized):
    default_transform = torchvision.transforms.Compose(
        [TRANSFORM_DEQUANTIZE_8BIT, transforms.Grayscale(num_output_channels=3)]
    )



class RenderedSST2Quantized(TorchVisionDataset):
    _data_source = torchvision.datasets.RenderedSST2
    _split_args = {TRAIN_SPLIT: {"split": 'train'}, VAL_SPLIT: {"split": 'val'}, TEST_SPLIT: {"split": 'test'}}
    # default_transform = torchvision.transforms.ToTensor()
    root_subdir = "RenderedSST2"
    default_transform = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(size=[32,32]), torchvision.transforms.ToTensor()])  # Shape [N, 32, 32, 3]


class RenderedSST2Dequantized(RenderedSST2Quantized):
    default_transform = TRANSFORM_DEQUANTIZE_8BIT


class RenderedSST2DequantizedGrey(RenderedSST2Quantized):
    default_transform = torchvision.transforms.Compose(
        [TRANSFORM_DEQUANTIZE_8BIT, transforms.Grayscale(num_output_channels=3)]
    )



class EuroSATQuantized(TorchVisionDataset):
    _data_source = torchvision.datasets.EuroSAT
    _split_args = {TRAIN_SPLIT: {"split": 'train'}, VAL_SPLIT: {"split": 'val'}, TEST_SPLIT: {"split": 'test'}}
    # default_transform = torchvision.transforms.ToTensor()
    root_subdir = "EuroSAT"
    default_transform = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(size=32), torchvision.transforms.ToTensor()])  # Shape [N, 32, 32, 3]
    # default_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(size=32), torchvision.transforms.ToTensor()])  # Shape [N, 32, 32, 3]


class EuroSATDequantized(EuroSATQuantized):
    default_transform = TRANSFORM_DEQUANTIZE_8BIT


class EuroSATDequantizedGrey(EuroSATQuantized):
    default_transform = torchvision.transforms.Compose(
        [TRANSFORM_DEQUANTIZE_8BIT, transforms.Grayscale(num_output_channels=3)]
    )


class DTDQuantized(TorchVisionDataset):
    _data_source = torchvision.datasets.DTD
    _split_args = {TRAIN_SPLIT: {"split": 'train'}, VAL_SPLIT: {"split": 'val'}, TEST_SPLIT: {"split": 'test'}}
    # default_transform = torchvision.transforms.ToTensor()
    root_subdir = "DTD"
    # default_transform = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(size=32), torchvision.transforms.ToTensor()])  # Shape [N, 32, 32, 3]
    default_transform = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(size=32), torchvision.transforms.ToTensor()])  # Shape [N, 32, 32, 3]


class DTDDequantized(DTDQuantized):
    default_transform = TRANSFORM_DEQUANTIZE_8BIT


class DTDDequantizedGrey(DTDQuantized):
    default_transform = torchvision.transforms.Compose(
        [TRANSFORM_DEQUANTIZE_8BIT, transforms.Grayscale(num_output_channels=3)]
    )

