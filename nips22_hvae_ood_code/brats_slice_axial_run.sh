#!/bin/sh

CUDA_VISIBLE_DEVICES=1 python dvae_run.py \
--epochs 6000 \
--batch_size 32 \
--free_nats 0 \
--free_nats_epochs 0 \
--warmup_epochs 0 \
--save-dir ./mood_models/ \
--test_every 1000 \
--train_importance_weighted \
--test_importance_weighted \
--train_datasets \
'{
"BratsSubvolumeLinearNormalization": {"split": "train", "dset_path": "/home/derek/mood_patches/brats_patches/brats_train_half_slices_reg_norm_no_scale.npz"}
}' \
--val_datasets \
'{
    "BratsSubvolumeLinearNormalization": {"split": "validation", "dset_path": "/home/derek/mood_patches/brats_patches/brats_train_half_slices_reg_norm_no_scale.npz", "toy": false, "anom": false},
    "BratsSubvolumeLinearNormalization": {"split": "validation", "dset_path": "/home/derek/mood_patches/brats_patches/brats_anom_half_slices_reg_norm_no_scale.npz", "toy": false, "anom": true}
}' \
--model VAE \
--likelihood GrayscaleContinuousLogisticMixLikelihoodConv2d \
--config_deterministic \
'[
    [
        {"block": "ResBlockConv2d", "out_channels": 64, "kernel_size": 5, "stride": 1, "weightnorm": true, "gated": false},
        {"block": "ResBlockConv2d", "out_channels": 64, "kernel_size": 5, "stride": 1, "weightnorm": true, "gated": false},
        {"block": "ResBlockConv2d", "out_channels": 64, "kernel_size": 5, "stride": 2, "weightnorm": true, "gated": false}
    ],
    [
        {"block": "ResBlockConv2d", "out_channels": 64, "kernel_size": 3, "stride": 1, "weightnorm": true, "gated": false},
        {"block": "ResBlockConv2d", "out_channels": 64, "kernel_size": 3, "stride": 1, "weightnorm": true, "gated": false},
        {"block": "ResBlockConv2d", "out_channels": 64, "kernel_size": 3, "stride": 2, "weightnorm": true, "gated": false}
    ],
    [
        {"block": "ResBlockConv2d", "out_channels": 64, "kernel_size": 3, "stride": 1, "weightnorm": true, "gated": false},
        {"block": "ResBlockConv2d", "out_channels": 64, "kernel_size": 3, "stride": 1, "weightnorm": true, "gated": false},
        {"block": "ResBlockConv2d", "out_channels": 64, "kernel_size": 3, "stride": 2, "weightnorm": true, "gated": false}
    ],
    [
        {"block": "ResBlockConv2d", "out_channels": 64, "kernel_size": 3, "stride": 1, "weightnorm": true, "gated": false},
        {"block": "ResBlockConv2d", "out_channels": 64, "kernel_size": 3, "stride": 1, "weightnorm": true, "gated": false},
        {"block": "ResBlockConv2d", "out_channels": 64, "kernel_size": 3, "stride": 1, "weightnorm": true, "gated": false}
    ]
]' \
--config_stochastic \
'[
    {"block": "GaussianConv2d", "latent_features": 32, "weightnorm": true},
    {"block": "GaussianConv2d", "latent_features": 16, "weightnorm": true},
    {"block": "GaussianConv2d", "latent_features": 8, "weightnorm": true},
    {"block": "GaussianDense", "latent_features": 4, "weightnorm": true}
]'
