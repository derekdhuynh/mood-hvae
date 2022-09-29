"""
Predict sample and pixel level scores for anomalies

TODO: Make sure there's no redundancy in compute_llr_scores function
"""
import os
from collections import defaultdict

import nibabel as nib
import numpy as np
import torch
import torchvision
from tqdm import trange

import oodd
import oodd.losses
import oodd.models
import oodd.datasets
import oodd.utils

def image_generator(img_paths, tol=0):
    """Loads numpy arrays of images from disk"""
    for path in img_paths:
        img = nib.load(path)
        img_data = img.get_fdata()
        yield img_data, tol

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
        
def make_strided_patches(img, patch_size=(28, 28), stride=(4, 4), bg_tol=0.05, tol=0.):
    """
    Given a 2D slice, create a dataset of by striding over the image and taking
    patches of size patch_size

    TODO: Add edge cases for strides that need padding
    """
    # All background voxels
    if (img <= tol).all() == True:
        return

    # Preprocessing, using contrast stretching
    pipeline = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        oodd.datasets.transforms.PctileChauhan(0.05)
    ])

    def preprocess(arr, pipeline):
        arr = arr.astype('float32')
        return pipeline(arr)
    
    patches = []
    coords = []
    
    x_stride = stride[0]
    y_stride = stride[1]
    
    x_max = img.shape[0]
    y_max = img.shape[1]
    
    n_xstrides = (x_max - patch_size[0]) // x_stride
    n_ystrides = (y_max - patch_size[1]) // y_stride
    for x_stride_num in range(n_xstrides + 1):
        x_cur = x_stride_num * x_stride
        for y_stride_num in range(n_ystrides + 1):
            y_cur = y_stride_num * y_stride
            y2 = y_cur + patch_size[1]
            
            x2 = x_cur + patch_size[0]
            patch = img[x_cur : x2, y_cur: y2]

            # Ignore patches with background more than bg_tol%
            if (patch <= tol).sum() < int(patch_size[0] * patch_size[0] * bg_tol):
                patches.append(preprocess(patch, pipeline))
                coords.append([x_cur, x2, y_cur, y2])
                
    if not patches:
        return

    patches = torch.cat(patches).view(-1, 1, patch_size[0], patch_size[1])
    coords = torch.tensor(coords)
    return patches, coords

def compute_llr_scores(img, model, iw_samples_elbo=5, iw_samples_Lk=1, n_latents_skip=2, batch_size=256, stride=(4, 4), tol=0., plane='axial'):
    device = torch.device('cuda')
    criterion = oodd.losses.ELBO()

    img_shape = img.shape
    abnormal_score_array = np.full(img_shape, 0., dtype='float32')
    mean_array = np.full(img_shape, 0., dtype='float32')
    patch_scores = []

    for z in trange(img_shape[-1]):
        if plane == 'axial':
            strided_patches = make_strided_patches(img[:, :, z], stride=stride, tol=tol)
        elif plane == 'sagittal':
            strided_patches = make_strided_patches(img[z, :, :], stride=stride, tol=tol)
        elif plane == 'coronal':
            strided_patches = make_strided_patches(img[:, z, :], stride=stride, tol=tol)

        if not strided_patches:
            continue
        else:
            stride_gen, coords = strided_patches

        dataset = torch.utils.data.TensorDataset(stride_gen, coords)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        decode_from_p = get_decode_from_p(model.n_latents, k=n_latents_skip)

        scores = [] 
        #elbos = [] 
        #elbos_k = [] 
        with torch.no_grad():
            n = 0
            for b, (x, coord) in enumerate(loader):
                x = x.to(device)
                n += x.shape[0]
                sample_elbos, sample_elbos_k = [], []

                # Regular ELBO
                for i in range(iw_samples_elbo):
                    likelihood_data, stage_datas = model(
                            x, decode_from_p=False, use_mode=False)
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
                        beta=1,
                        sample_reduction=None,
                        batch_reduction=None,
                    )
                    sample_elbos.append(elbo.detach())

                # L>k bound
                for i in range(iw_samples_Lk):
                    likelihood_data_k, stage_datas_k = model(
                            x, decode_from_p=decode_from_p, 
                            use_mode=decode_from_p
                    )
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
                    sample_elbos_k.append(elbo_k.detach())

                sample_elbos = torch.stack(sample_elbos, axis=0)
                sample_elbos_k = torch.stack(sample_elbos_k, axis=0)

                sample_elbo = oodd.utils.log_sum_exp(sample_elbos, axis=0)
                sample_elbo_k = oodd.utils.log_sum_exp(sample_elbos_k, axis=0)

                score = sample_elbo - sample_elbo_k
                
                for ind, (x1, x2, y1, y2) in enumerate(coord):
                    if plane == 'axial':
                        abnormal_score_array[x1:x2, y1:y2, z] += score[ind].detach().cpu().numpy()
                        mean_array[x1:x2, y1:y2, z] += 1
                    elif plane == 'sagittal':
                        abnormal_score_array[z, x1:x2, y1:y2] += score[ind].detach().cpu().numpy()
                        mean_array[z, x1:x2, y1:y2] += 1
                    elif plane == 'coronal':
                        abnormal_score_array[x1:x2, z, y1:y2] += score[ind].detach().cpu().numpy()
                        mean_array[x1:x2, z, y1:y2] += 1

                    #abnormal_score_array[x1:x2, y1:y2, z] += score[ind].detach().cpu().numpy()
                    #mean_array[x1:x2, y1:y2, z] += 1

                scores.extend(score.tolist())
                #elbos.extend(sample_elbo.tolist())
                #elbos_k.extend(sample_elbo_k.tolist())
        patch_scores.append(scores)

    patch_scores = np.concatenate(patch_scores)
    return abnormal_score_array, mean_array, patch_scores


def predict_folder_pixel_abs(input_folder, target_folder):
    sample_img = os.path.join(input_folder, os.listdir(input_folder)[0])

    sample_img_shape = nib.load(sample_img).shape

    if sample_img_shape[0] == 512:
        stride = (11, 11)
        iw_samples_elbo = 1
        tol = 0.02
        models = {
            'axial': '/workspace/models/abdom/mood_abdom_axial_10-mix_500-epochs_no-cool_no-nats',
            'sagittal': None,
            'coronal': None
        }

        for k in models:
            if not models[k]:
                continue
            path = models[k]
            models[k] = oodd.models.Checkpoint(path=path)

            # Loading models
            models[k].load_model()
            models[k] = models[k].model
            models[k].eval()
            models[k].to(device)

        # Thresholds for normalizing scores to [0, 1]
        thresholds = {'axial': [6000, 7850], 'sagittal': [6000, 7950], 'coronal': [6000, 7750]}

        #mn = 7000
        #mx = 7900
        
    else:
        stride = (4, 4)
        iw_samples_elbo = 1
        tol = 0.
        models = {
            'axial': '/workspace/models/brain/mood_brain_axial_10-mix_1000-epochs_no-cool_no-nats',
            'sagittal': '/workspace/models/brain/mood_brain_sagittal_10-mix_1000-epochs_no-cool_no-nats',
            'coronal': '/workspace/models/brain/mood_brain_coronal_10-mix_1000-epochs_no-cool_no-nats'
        }
        for k in models:
            path = models[k]
            models[k] = oodd.models.Checkpoint(path=path)

            # Loading models
            models[k].load_model()
            models[k] = models[k].model
            models[k].eval()
            models[k].to(device)

        # Thresholds for normalizing scores to [0, 1]
        thresholds = {'axial': [6000, 7850], 'sagittal': [6000, 7950], 'coronal': [6000, 7750]}
        #mn = 6000
        #mx = 7900

    for f in os.listdir(input_folder):
        source_file = os.path.join(input_folder, f)
        target_file = os.path.join(target_folder, f)

        nimg = nib.load(source_file)
        nimg_array = nimg.get_fdata()

        voxel_scores = []
        for plane in models:
            model = models[plane]

            if not model:
                continue

            abnormal_score_array, mean_array, _ = compute_llr_scores(nimg_array, model, iw_samples_elbo=iw_samples_elbo, iw_samples_Lk=1, stride=stride, tol=tol, plane=plane)

            abnormal_score_array /= mean_array
            abnormal_score_array = np.where(np.isnan(abnormal_score_array), 0, abnormal_score_array)

            mn = thresholds[plane][0]
            mx = thresholds[plane][1]

            normalized = np.where(abnormal_score_array < mn, 0, abnormal_score_array)
            normalized = np.where(abnormal_score_array > mx, 1, normalized)
            normalized = np.where((abnormal_score_array < mx) & (abnormal_score_array > mn), (abnormal_score_array-mn)/(mx-mn), normalized)
            voxel_scores.append(normalized)

        combined_scores = np.mean(voxel_scores, axis=0) #* (nimg_array > tol)

        final_nimg = nib.Nifti1Image(combined_scores, affine=nimg.affine)
        nib.save(final_nimg, target_file)

def predict_folder_sample_abs(input_folder, target_folder):
    sample_img = os.path.join(input_folder, os.listdir(input_folder)[0])
    sample_img_shape = nib.load(sample_img).shape

    if sample_img_shape[0] == 512:
        stride = (11, 11)
        iw_samples_elbo = 1
        tol = 0.02
        checkpoint = oodd.models.Checkpoint(path='/workspace/models/abdom/mood_abdom_axial_10-mix_500-epochs_no-cool_no-nats')
        mn = 7800
        mx = 7900
        
    else:
        stride = (4, 4)
        iw_samples_elbo = 5
        tol = 0.
        checkpoint = oodd.models.Checkpoint(path='/workspace/models/brain/mood_brain_axial_10-mix_1000-epochs_no-cool_no-nats')
        mn = 7800
        mx = 7900

    # Loading model
    checkpoint.load_model()
    model = checkpoint.model
    model.eval()
    model.to(device)

    for f in os.listdir(input_folder):
        source_file = os.path.join(input_folder, f)

        nimg = nib.load(source_file)
        nimg_array = nimg.get_fdata()

        _, _, patch_scores = compute_llr_scores(nimg_array, model, 
                iw_samples_elbo=iw_samples_elbo, stride=stride, tol=tol)

        if np.max(patch_scores) > mx:
            abnormal_score = 1
        elif np.max(patch_scores) < mn:
            abnormal_score = 0
        else:
            abnormal_score = (np.max(patch_scores) - mn) / (mx - mn)

        with open(os.path.join(target_folder, f + ".txt"), "w") as write_file:
            write_file.write(str(abnormal_score))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("-o", "--output", required=True, type=str)
    parser.add_argument("-mode", type=str, default="pixel", help="can be either 'pixel' or 'sample'.", required=False)

    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    mode = args.mode

    device = torch.device('cuda')

    # Loading model
    criterion = oodd.losses.ELBO()

    scores_dict = defaultdict(list)

    if mode == "pixel":
        predict_folder_pixel_abs(input_dir, output_dir)
    elif mode == "sample":
        predict_folder_sample_abs(input_dir, output_dir)
    else:
        print("Mode not correctly defined. Either choose 'pixel' or 'sample'")
