import numpy as np
import torch
import torch.nn as nn

class PctileChauhan(nn.Module):
    def __init__(self, pct):
        super().__init__()
        self.pct = pct

    def forward(self, x):
        # Flatten along dims if image has multiple channels
        img = x
        dim = None
        keepdim = False
        if x.shape[0] > 1:
            img = x.view(-1, x.shape[1] * x.shape[2])
            dim = 1
            keepdim = False

        x_top_pct = torch.quantile(img, 1. - self.pct, interpolation='nearest', dim=dim, keepdim=keepdim)
        x_bottom_pct = torch.quantile(img, self.pct, interpolation='nearest', dim=dim, keepdim=keepdim) 

        # If most of the image is constant, then top_pct and bottom_pct will
        # be the same, causing divide by 0
        same_div = x_top_pct == x_bottom_pct
        if same_div.any():
            x_top_pct = torch.where(same_div, torch.quantile(img, 1., interpolation='nearest', dim=dim, keepdim=keepdim), x_top_pct)
            x_bottom_pct = torch.where(same_div, torch.quantile(img, 0., interpolation='nearest', dim=dim, keepdim=keepdim), x_bottom_pct)

            #x_top_pct = torch.quantile(img, 1., interpolation='nearest', dim=dim)
            #x_bottom_pct = torch.quantile(img, 0., interpolation='nearest', dim=dim)

        # Edge case for all black image
        all_black = (x_top_pct == 0.).any()
        all_constant = (x_top_pct == x_bottom_pct).any()
        if all_black:
            x_top_pct = torch.where(all_black, torch.tensor(1.).float(), x_top_pct)

        # Edge case for if the entire image is constant
        elif all_constant:
            x_bottom_pct = torch.where(all_constant, torch.tensor(0.).float(), x_bottom_pct)

        x = x.permute(1, 2, 0)
        return torch.clip((x - x_bottom_pct) / (x_top_pct - x_bottom_pct), 0., 1.).permute(2, 0, 1)

class PctileKevin(nn.Module):
    def __init__(self, pct):
        super().__init__()
        self.pct = pct
    
    def forward(self, x):
        nth_pctile = torch.quantile(x, self.pct, interpolation='nearest')
        return torch.where(x > nth_pctile, nth_pctile, x) / nth_pctile


class AsFloatTensor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.type(torch.FloatTensor)


class Scale(nn.Module):
    def __init__(self, a=None, b=None, min_val=None, max_val=None):
        """Scale an input to be in [a, b] by normalizing with data min and max values"""
        super().__init__()
        assert (a is not None) == (b is not None), "must set both a and b or neither"
        self.a = a
        self.b = b
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        x_min = self.min_val if self.min_val is not None else x.min()
        x_max = self.max_val if self.max_val is not None else x.max()
        x_scaled = (x - x_min) / (x_max - x_min)
        if self.a is None:
            return x_scaled

        return self.a + x_scaled * (self.b - self.a)


class Binarize(nn.Module):
    def __init__(self, resample: bool = False, threshold: float = None):
        super().__init__()
        assert bool(threshold) != bool(resample), "Must set exactly one of threshold and resample"
        self.resample = resample
        self.threshold = threshold

    def forward(self, x):
        if self.resample:
            return torch.bernoulli(x)

        return x > self.threshold


class Dequantize(nn.Module):
    """Dequantize a quantized data point by adding uniform noise.

    Sppecifically, assume the quantized data is x in {0, 1, 2, ..., D} for some D e.g. 255 for int8 data.
    Then, the transformation is given by definition of the dequantized data z as

        z = x + u
        u ~ U(0, 1)

    where u is sampled uniform noise of same shape as x.

    The dequantized data is in the continuous interval [0, D + 1]

    If the value is to scaled subsequently, the maximum value attainable is hence D + 1 due to the uniform noise.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + torch.rand_like(x)


class InvertGrayScale(nn.Module):
    def __init__(self, max_val=1):
        """Invert a gray-scale image in [0, 1] by flipping the colour scale such that 0 becomes 1 and 1 becomes 0."""
        super().__init__()
        self.max_val = max_val

    def forward(self, x):
        return self.max_val - x


class Grayscale(nn.Module):
    """Convert tensor to grey scale.

    See https://www.kdnuggets.com/2019/12/convert-rgb-image-grayscale.html
    """

    def __init__(self, num_output_channels=1):
        super().__init__()
        self.num_output_channels = num_output_channels

    @staticmethod
    def luminance_perception_correction(x):
        """Given an RGB input of dimensions (C, H, W), return the luminance corrected grey-scale version as (H, W)"""
        return 0.2126 * x[0] + 0.7152 * x[1] + 0.0722 * x[2]

    @staticmethod
    def gamma_expansion(x):
        """Correct the "channel-less" luminance perception corrected grey-scale image with inverse gamma compression"""
        idx = x <= 0.04045
        x[idx] = x[idx] / 12.92
        x[~idx] = ((x[~idx] + 0.055) / 1.055) ** 2.4
        return x

    @staticmethod
    def expand_channels(x, num_channels):
        """Expand the (H, W) shaped image to have 'num_channels' before the HW dimensions, i.e. (C, H, W)"""
        return np.stack(
            (x,) * num_channels, axis=0
        )

    def forward(self, x):
        x = self.luminance_perception_correction(x)
        x = self.gamma_expansion(x)
        return self.expand_channels(x, self.num_output_channels)


class Permute(nn.Module):
    def __init__(self, *dims):
        """Permute the dims of a tensor similar to https://pytorch.org/docs/stable/tensors.html#torch.Tensor.permute"""
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)
