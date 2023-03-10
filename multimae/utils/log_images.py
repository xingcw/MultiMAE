# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import wandb

import multimae.utils as utils
from pipelines.utils.constants import UNITY_COARSE_SEM_LABELS, IMG_HEIGHT, IMG_WIDTH
from multimae.utils.data_constants import CUSTOM_SEMSEG_NUM_CLASSES
from multimae.utils.datasets_semseg import (ade_classes, hypersim_classes, nyu_v2_40_classes)
from multimae.utils.plot_utils import plot_predictions


def inv_norm(tensor: torch.Tensor) -> torch.Tensor:
    """Inverse of the normalization that was done during pre-processing
    """
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

    return inv_normalize(tensor)

@torch.no_grad()
def log_multimae_semseg_wandb(
    inputs: Dict[str, torch.Tensor],
    preds: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
    image_count=3,
    prefix: str = "",
    metadata = None
):
    log_images = {}
    class_labels = UNITY_COARSE_SEM_LABELS  
    common_domains = list(set(inputs.keys()).intersection(set(preds.keys())))
    
    if len(common_domains) > 0:
        image_count = min(len(inputs[common_domains[0]]), image_count)
        for i in range(image_count):
            common_inputs = {k: inputs[k][i].unsqueeze(0) for k in common_domains}
            common_preds = {k: preds[k][i].unsqueeze(0) for k in common_domains}
            common_masks = {k: masks[k][i].unsqueeze(0) for k in common_domains}
            unify_plot = plot_predictions(common_inputs, common_preds, common_masks, 
                                          show_img=False, metadata=metadata, return_fig=True)
            log_images[f"{prefix}_compare_{i}"] = wandb.Image(unify_plot)
    
    rgb_imgs = inputs["rgb"]
    image_count = min(len(rgb_imgs), image_count)
    depth_gts = inputs["depth"] if "depth" in preds else None
    semseg_gts = inputs["semseg"] if "semseg" in preds else []
    semseg_preds = preds["semseg"] if "semseg" in preds else []
    
    rgb_imgs = rgb_imgs[:image_count]
    semseg_preds = semseg_preds[:image_count]
    semseg_gts = semseg_gts[:image_count]
    depth_gts = depth_gts[:image_count] if depth_gts is not None else None
    depth_gts = depth_gts.cpu().numpy() if depth_gts is not None else None
    
    # rescale semseg predictions
    seg_pred_argmax = semseg_preds[:, :CUSTOM_SEMSEG_NUM_CLASSES].argmax(dim=1)
    image_size = (IMG_WIDTH, IMG_HEIGHT)
    semseg_preds = F.interpolate(seg_pred_argmax.unsqueeze(1).float(), size=image_size, mode='nearest').squeeze(1)
    semseg_gts = F.interpolate(semseg_gts.unsqueeze(1).float(), size=image_size, mode='nearest').squeeze(1)
    semseg_gts, semseg_preds = semseg_gts.cpu().numpy(), semseg_preds.cpu().numpy()

    for i, (image, pred, gt) in enumerate(zip(rgb_imgs, semseg_preds, semseg_gts)):
               
        image = inv_norm(image)
         
        semseg_image = wandb.Image(image, masks={
            "predictions": {
                "mask_data": pred,
                "class_labels": class_labels,
            },
            "ground_truth": {
                "mask_data": gt,
                "class_labels": class_labels,
            }
        })

        log_images[f"{prefix}_semseg_{i}"] = semseg_image

        if depth_gts is not None:
            log_images[f"{prefix}_depth_{i}"] = wandb.Image(depth_gts[i])
            
    wandb.log(log_images, commit=False)


@torch.no_grad()
def log_semseg_wandb(
        images: torch.Tensor, 
        preds: List[np.ndarray], 
        gts: List[np.ndarray],
        depth_gts: List[np.ndarray],
        dataset_name: str = 'ade20k',
        image_count=8, 
        prefix=""
    ):

    if dataset_name == 'ade20k':
        classes = ade_classes()
    elif dataset_name == 'hypersim':
        classes = hypersim_classes()
    elif dataset_name == 'nyu':
        classes = nyu_v2_40_classes()
    else:
        raise ValueError(f'Dataset {dataset_name} not supported for logging to wandb.')

    class_labels = {i: cls for i, cls in enumerate(classes)}
    class_labels[len(classes)] = "void"
    class_labels[utils.SEG_IGNORE_INDEX] = "ignore"

    image_count = min(len(images), image_count)

    images = images[:image_count]
    preds = preds[:image_count]
    gts = gts[:image_count]
    depth_gts = depth_gts[:image_count] if len(depth_gts) > 0 else None

    semseg_images = {}

    for i, (image, pred, gt) in enumerate(zip(images, preds, gts)):
        image = inv_norm(image)
        pred[gt == utils.SEG_IGNORE_INDEX] = utils.SEG_IGNORE_INDEX

        semseg_image = wandb.Image(image, masks={
            "predictions": {
                "mask_data": pred,
                "class_labels": class_labels,
            },
            "ground_truth": {
                "mask_data": gt,
                "class_labels": class_labels,
            }
        })

        semseg_images[f"{prefix}_{i}"] = semseg_image

        if depth_gts is not None:
            semseg_images[f"{prefix}_{i}_depth"] = wandb.Image(depth_gts[i])

    wandb.log(semseg_images, commit=False)


@torch.no_grad()
def log_taskonomy_wandb(
        preds: Dict[str, torch.Tensor], 
        gts: Dict[str, torch.Tensor], 
        image_count=8, 
        prefix=""
    ):
    pred_tasks = list(preds.keys())
    gt_tasks = list(gts.keys())
    if 'mask_valid' in gt_tasks:
        gt_tasks.remove('mask_valid')

    image_count = min(len(preds[pred_tasks[0]]), image_count)

    all_images = {}

    for i in range(image_count):

        # Log GTs
        for task in gt_tasks:
            gt_img = gts[task][i]
            if task == 'rgb':
                gt_img = inv_norm(gt_img)
            if gt_img.shape[0] == 1:
                gt_img = gt_img[0]
            elif gt_img.shape[0] == 2:
                gt_img = F.pad(gt_img, (0,0,0,0,0,1), mode='constant', value=0.0)

            gt_img = wandb.Image(gt_img, caption=f'GT #{i}')
            key = f'{prefix}_gt_{task}'
            if key not in all_images:
                all_images[key] = [gt_img]
            else:
                all_images[key].append(gt_img)

        # Log preds
        for task in pred_tasks:
            pred_img = preds[task][i]
            if task == 'rgb':
                pred_img = inv_norm(pred_img)
            if pred_img.shape[0] == 1:
                pred_img = pred_img[0]
            elif pred_img.shape[0] == 2:
                pred_img = F.pad(pred_img, (0,0,0,0,0,1), mode='constant', value=0.0)

            pred_img = wandb.Image(pred_img, caption=f'Pred #{i}')
            key = f'{prefix}_pred_{task}'
            if key not in all_images:
                all_images[key] = [pred_img]
            else:
                all_images[key].append(pred_img)

    wandb.log(all_images, commit=False)
