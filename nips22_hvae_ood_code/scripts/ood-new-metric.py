import argparse
import os
import logging

from collections import defaultdict
from typing import *

from tqdm import tqdm

import rich
import numpy as np
import torch

import oodd
import oodd.datasets
import oodd.evaluators
import oodd.models
import oodd.losses
import oodd.utils


LOGGER = logging.getLogger()


parser = argparse.ArgumentParser()


trained_model_dir = "vae-dc-cifar"
importance_samples = 1
trained_on = 'cifar'  # 'fmnist'
parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument("--model_dir", type=str, default="./models/"+trained_model_dir, help="model")  # best: im1 -> celeba 0.723; im101:celeba 0.7265
parser.add_argument("--save_dir", type=str, default="./results/"+trained_model_dir, help="directory to store scores in")
parser.add_argument("--iw_samples_elbo", type=int, default=importance_samples, help="importances samples for regular ELBO")
parser.add_argument("--iw_samples_Lk", type=int, default=importance_samples, help="importances samples for L>k bound")
parser.add_argument("--n_eval_examples", type=int, default=float("inf"), help="cap on the number of examples to use")
parser.add_argument("--n_latents_skip", type=int, default=2, help="the value of k in the paper")
parser.add_argument("--batch_size", type=int, default=100, help="batch size for evaluation")
parser.add_argument("--device", type=str, default="cuda:3", help="device to evaluate on")


args = parser.parse_args()
rich.print(vars(args))

os.makedirs(args.save_dir, exist_ok=True)
device = oodd.utils.get_device() if args.device == "auto" else torch.device(args.device)
LOGGER.info("Device %s", device)

FILE_NAME_SETTINGS_SPEC = f"k{args.n_latents_skip}-iw_elbo{args.iw_samples_elbo}-iw_lK{args.iw_samples_Lk}"


def get_save_path(name):
    name = name.replace(" ", "-")
    return f"{args.save_dir}/{name}"


def get_decode_from_p(n_latents, k=0, semantic_k=True):
    """
    k semantic out
    0 True     [False, False, False]
    1 True     [True, False, False]
    2 True     [True, True, False]
    0 False    [True, True, True]
    1 False    [False, True, True]
    2 False    [False, False, True]
    """
    if semantic_k:
        return [True] * k + [False] * (n_latents - k)

    return [False] * (k + 1) + [True] * (n_latents - k - 1)


def get_lengths(dataloaders):
    return [len(loader) for name, loader in dataloaders.items()]


def print_stats(llr, l, lk):
    llr_mean, llr_var, llr_std = np.mean(llr), np.var(llr), np.std(llr)
    l_mean, l_var, l_std = np.mean(l), np.var(l), np.std(l)
    lk_mean, lk_var, lk_std = np.mean(lk), np.var(lk), np.std(lk)
    s = f"  {l_mean:8.3f},   {l_var:8.3f},   {l_std:8.3f}\n"
    s += f"{llr_mean:8.3f}, {llr_var:8.3f}, {llr_std:8.3f}\n"
    s += f" {lk_mean:8.3f},  {lk_var:8.3f},  {lk_std:8.3f}"
    print(s)


# Define checkpoints and load model
checkpoint = oodd.models.Checkpoint(path=args.model_dir)
checkpoint.load(device=device)
datamodule = checkpoint.datamodule
model = checkpoint.model
model.eval()
criterion = oodd.losses.ELBO()
rich.print(datamodule)

# Add additional datasets to evaluation
TRAIN_DATASET_KEY = list(datamodule.train_datasets.keys())[0]
LOGGER.info("Train dataset %s", TRAIN_DATASET_KEY)

MAIN_DATASET_NAME = datamodule.primary_val_name.strip("Binarized").strip("Quantized").strip("Dequantized")
LOGGER.info("Main dataset %s", MAIN_DATASET_NAME)

IN_DIST_DATASET = MAIN_DATASET_NAME + " test"
TRAIN_DATASET = MAIN_DATASET_NAME + " train"
LOGGER.info("Main in-distribution dataset %s", IN_DIST_DATASET)
if MAIN_DATASET_NAME in ["FashionMNIST", "MNIST"]:
    extra_val = dict(
        notMNISTQuantized=dict(split='validation'),
    )
    extra_test = {TRAIN_DATASET_KEY: dict(split="train", dynamic=False)}
elif MAIN_DATASET_NAME in ["CIFAR10", "SVHN"]:
    extra_val = dict(
        # LFWPeopleQuantized=dict(split='train'),  # 人脸
    )
    extra_test = {TRAIN_DATASET_KEY: dict(split="train", dynamic=True)}
else:
    raise ValueError(f"Unknown main dataset name {MAIN_DATASET_NAME}")

datamodule.add_datasets(val_datasets=extra_val, test_datasets=extra_test)
datamodule.data_workers = 4
datamodule.batch_size = args.batch_size
datamodule.test_batch_size = args.batch_size
LOGGER.info("%s", datamodule)


n_test_batches = get_lengths(datamodule.val_datasets) + get_lengths(datamodule.test_datasets)
for name, loader in datamodule.val_datasets.items():
    print(f'dataset:{name}-->len:{len(loader)}')
N_EQUAL_EXAMPLES_CAP = int(min(n_test_batches)/1000)*1000
print(f'N_EQUAL_EXAMPLES_CAP:{N_EQUAL_EXAMPLES_CAP}')
assert N_EQUAL_EXAMPLES_CAP % args.batch_size == 0, "Batch size must divide smallest dataset size"
if trained_on == 'fmnist':
    args.beta = 0.001

N_EQUAL_EXAMPLES_CAP = min([args.n_eval_examples, N_EQUAL_EXAMPLES_CAP])
LOGGER.info("%s = %s", "N_EQUAL_EXAMPLES_CAP", N_EQUAL_EXAMPLES_CAP)

decode_from_p = get_decode_from_p(model.n_latents, k=args.n_latents_skip)

dataloaders = {(k + " test", v) for k, v in datamodule.val_loaders.items()}
dataloaders |= {(k + " train", v) for k, v in datamodule.test_loaders.items()}

scores = defaultdict(list)
scores_2 = defaultdict(list)
scores_3 = defaultdict(list)
elbos = defaultdict(list)
elbos_k = defaultdict(list)
with torch.no_grad():
    for dataset, dataloader in dataloaders:
        dataset = dataset.replace("Binarized", "").replace("Quantized", "").replace("Dequantized", "")
        print(f"Evaluating {dataset}")

        n = 0
        for b, (x, _) in tqdm(enumerate(dataloader), total=N_EQUAL_EXAMPLES_CAP / args.batch_size):
            x = x.to(device)

            n += x.shape[0]
            sample_elbos, sample_elbos_k, sample_recon_likes = [], [], []

            # Regular ELBO
            for i in tqdm(range(args.iw_samples_elbo), leave=False):
                likelihood_data, stage_datas, _ = model(x, decode_from_p=False, use_mode=False)
                kl_divergences = [
                    stage_data.loss.kl_elementwise
                    for stage_data in stage_datas
                    if stage_data.loss.kl_elementwise is not None
                ]
                loss, elbo, likelihood, kl_divergences = criterion(
                    likelihood_data.likelihood,
                    kl_divergences,
                    samples=1,
                    free_nats=0,
                    beta=args.beta,
                    sample_reduction=None,
                    batch_reduction=None,
                )
                sample_elbos.append(elbo.detach())
                sample_recon_likes.append(likelihood)
                # sample_elbos.append(-loss.detach())

            llrs = [[]for _ in range(model.n_latents)]
            skip_elbos = [[]for _ in range(model.n_latents)]
            skip_likes = [[]for _ in range(model.n_latents)]
            llrs[0].append(0)
            skip_elbos[0].append(elbo.detach())
            skip_likes[0].append(likelihood.detach())
            sample_elbos = torch.stack(sample_elbos, axis=0)
            sample_recon_likes = torch.stack(sample_recon_likes, axis=0)
            for skip_latents in range(1, model.n_latents):
                decode_from_p = get_decode_from_p(model.n_latents, k=skip_latents)
                # L>k bound
                for i in tqdm(range(args.iw_samples_Lk), leave=False):
                    likelihood_data_k, stage_datas_k, _ = model(x, decode_from_p=decode_from_p, use_mode=decode_from_p)
                    kl_divergences_k = [
                        stage_data.loss.kl_elementwise
                        for stage_data in stage_datas_k
                        if stage_data.loss.kl_elementwise is not None
                    ]
                    loss_k, elbo_k, likelihood_k, kl_divergences_k = criterion(
                        likelihood_data_k.likelihood,
                        kl_divergences_k,
                        samples=1,
                        free_nats=0,
                        beta=1,
                        sample_reduction=None,
                        batch_reduction=None,
                    )
                    skip_elbos[skip_latents].append(elbo_k.detach())
                    skip_likes[skip_latents].append(likelihood.detach())
                    # sample_elbos_k.append(elbo_k.detach())


                    sample_elbos_k = torch.stack(skip_elbos[skip_latents], axis=0)
                    sample_recon_likes = torch.stack(skip_likes[skip_latents], axis=0)

                    sample_elbo = oodd.utils.log_sum_exp(sample_elbos, axis=0)
                    sample_elbo_k = oodd.utils.log_sum_exp(sample_elbos_k, axis=0)


                    llrs[skip_latents].append(sample_elbo - sample_elbo_k)

            score_3 = llrs[1][0]
            for l in range(1, model.n_latents-1):
                score_3 += (llrs[l+1][0] - llrs[l][0]) * (skip_likes[l+1][0] / skip_likes[l][0])
            scores_3[dataset].extend(score_3.tolist())

            # compute new metric by score = llr1 + \sum_{l=1}^{L-2} [ (llr^{l+1} - llr^{l}) * (elbo^{l}/elbo^{l+1}) ]
            # score_2 = llrs[1][0]
            # for l in range(1, model.n_latents-1):
            #     score_2 += (llrs[l+1][0] - llrs[l][0]) * (skip_elbos[l][0] / skip_elbos[l+1][0])
            # scores_2[dataset].extend(score_2.tolist())


            # elbos[dataset].extend(sample_elbo.tolist())
            # elbos_k[dataset].extend(sample_elbo_k.tolist())


            if n > N_EQUAL_EXAMPLES_CAP:
                LOGGER.warning(f"Skipping remaining iterations due to {N_EQUAL_EXAMPLES_CAP}")
                break


# print likelihoods
for dataset in sorted(scores.keys()):
    print("===============", dataset, "===============")
    print_stats(scores[dataset], elbos[dataset], elbos_k[dataset])

# save scores
# torch.save(scores, get_save_path(f"values-scores_1-{IN_DIST_DATASET}-{FILE_NAME_SETTINGS_SPEC}.pt"))
# torch.save(scores_2, get_save_path(f"values-scores_2-{IN_DIST_DATASET}-{FILE_NAME_SETTINGS_SPEC}.pt"))
torch.save(scores_3, get_save_path(f"values-scores_3-{IN_DIST_DATASET}-{FILE_NAME_SETTINGS_SPEC}.pt"))
# torch.save(elbos_k, get_save_path(f"values-elbos_k-{IN_DIST_DATASET}-{FILE_NAME_SETTINGS_SPEC}.pt"))
print(f'N_EQUAL_EXAMPLES_CAP:{N_EQUAL_EXAMPLES_CAP}')