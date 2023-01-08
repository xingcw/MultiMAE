# To supress DPT and Mask2Former warnings
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# Mask2Former and detectron2 dependencies for semantic segmentation pseudo labeling
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from multimae.utils.data_constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def get_masked_image(img, mask, image_size=224, patch_size=16, mask_value=0.0):
    img_token = rearrange(
        img.detach().cpu(), 
        'b c (nh ph) (nw pw) -> b (nh nw) (c ph pw)', 
        ph=patch_size, pw=patch_size, nh=image_size//patch_size, nw=image_size//patch_size
    )
    img_token[mask.detach().cpu()!=0] = mask_value
    img = rearrange(
        img_token, 
        'b (nh nw) (c ph pw) -> b c (nh ph) (nw pw)', 
        ph=patch_size, pw=patch_size, nh=image_size//patch_size, nw=image_size//patch_size
    )
    return img


def denormalize(img, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    return TF.normalize(
        img.clone(),
        mean= [-m/s for m, s in zip(mean, std)],
        std= [1/s for s in std]
    )

def plot_semseg_gt(input_dict, ax=None, image_size=224):
    metadata = MetadataCatalog.get("coco_2017_val_panoptic")
    instance_mode = ColorMode.IMAGE
    img_viz = 255 * denormalize(input_dict['rgb'].detach().cpu())[0].permute(1,2,0)
    semseg = F.interpolate(
        input_dict['semseg'].unsqueeze(0).cpu().float(), size=image_size, mode='nearest'
    ).long()[0,0]
    visualizer = Visualizer(img_viz, metadata, instance_mode=instance_mode, scale=1)
    visualizer.draw_sem_seg(semseg)
    if ax is not None:
        ax.imshow(visualizer.get_output().get_image())
    else:
        return visualizer.get_output().get_image()


def plot_semseg_gt_masked(input_dict, mask, ax=None, mask_value=1.0, image_size=224):
    img = plot_semseg_gt(input_dict, image_size=image_size)
    img = torch.LongTensor(img).permute(2,0,1).unsqueeze(0)
    masked_img = get_masked_image(img.float()/255.0, mask, image_size=image_size, patch_size=16, mask_value=mask_value)
    masked_img = masked_img[0].permute(1,2,0)
    
    if ax is not None:
        ax.imshow(masked_img)
    else:
        return masked_img


def get_pred_with_input(gt, pred, mask, image_size=224, patch_size=16):
    gt_token = rearrange(
        gt.detach().cpu(), 
        'b c (nh ph) (nw pw) -> b (nh nw) (c ph pw)', 
        ph=patch_size, pw=patch_size, nh=image_size//patch_size, nw=image_size//patch_size
    )
    pred_token = rearrange(
        pred.detach().cpu(), 
        'b c (nh ph) (nw pw) -> b (nh nw) (c ph pw)', 
        ph=patch_size, pw=patch_size, nh=image_size//patch_size, nw=image_size//patch_size
    )
    pred_token[mask.detach().cpu()==0] = gt_token[mask.detach().cpu()==0]
    img = rearrange(
        pred_token, 
        'b (nh nw) (c ph pw) -> b c (nh ph) (nw pw)', 
        ph=patch_size, pw=patch_size, nh=image_size//patch_size, nw=image_size//patch_size
    )
    return img


def plot_semseg_pred_masked(rgb, semseg_preds, semseg_gt, mask, ax=None, image_size=224):
    metadata = MetadataCatalog.get("coco_2017_val_panoptic")
    instance_mode = ColorMode.IMAGE
    img_viz = 255 * denormalize(rgb.detach().cpu())[0].permute(1,2,0)
    
    semseg = get_pred_with_input(
        semseg_gt.unsqueeze(1), 
        semseg_preds.argmax(1).unsqueeze(1), 
        mask, 
        image_size=image_size//4, 
        patch_size=4
    )
    
    semseg = F.interpolate(semseg.float(), size=image_size, mode='nearest')[0,0].long()

    visualizer = Visualizer(img_viz, metadata, instance_mode=instance_mode, scale=1)
    visualizer.draw_sem_seg(semseg)
    if ax is not None:
        ax.imshow(visualizer.get_output().get_image())
    else:
        return visualizer.get_output().get_image()

def plot_predictions(input_dict, preds, masks, image_size=224):

    masked_rgb = get_masked_image(
        denormalize(input_dict['rgb']), 
        masks['rgb'],
        image_size=image_size,
        mask_value=1.0
    )[0].permute(1,2,0).detach().cpu()
    masked_depth = get_masked_image(
        input_dict['depth'], 
        masks['depth'],
        image_size=image_size,
        mask_value=np.nan
    )[0,0].detach().cpu()

    pred_rgb = denormalize(preds['rgb'])[0].permute(1,2,0).clamp(0,1)
    pred_depth = preds['depth'][0,0].detach().cpu()

    pred_rgb2 = get_pred_with_input(
        denormalize(input_dict['rgb']), 
        denormalize(preds['rgb']).clamp(0,1), 
        masks['rgb'],
        image_size=image_size
    )[0].permute(1,2,0).detach().cpu()
    pred_depth2 = get_pred_with_input(
        input_dict['depth'], 
        preds['depth'], 
        masks['depth'],
        image_size=image_size
    )[0,0].detach().cpu()

    fig = plt.figure(figsize=(10, 10))
    grid = ImageGrid(fig, 111, nrows_ncols=(3, 3), axes_pad=0)

    grid[0].imshow(masked_rgb)
    grid[1].imshow(pred_rgb2)
    grid[2].imshow(denormalize(input_dict['rgb'])[0].permute(1,2,0).detach().cpu())

    grid[3].imshow(masked_depth)
    grid[4].imshow(pred_depth2)
    grid[5].imshow(input_dict['depth'][0,0].detach().cpu())
    
    use_semseg = "semseg" in input_dict
    
    if use_semseg:
        plot_semseg_gt_masked(input_dict, masks['semseg'], grid[6], mask_value=1.0, image_size=image_size)
        plot_semseg_pred_masked(input_dict['rgb'], preds['semseg'], input_dict['semseg'], masks['semseg'], grid[7], image_size=image_size)
        plot_semseg_gt(input_dict, grid[8], image_size=image_size)

    for ax in grid:
        ax.set_xticks([])
        ax.set_yticks([])
                
    fontsize = 16
    grid[0].set_title('Masked inputs', fontsize=fontsize)
    grid[1].set_title('MultiMAE predictions', fontsize=fontsize)
    grid[2].set_title('Original Reference', fontsize=fontsize)
    grid[0].set_ylabel('RGB', fontsize=fontsize)
    grid[3].set_ylabel('Depth', fontsize=fontsize)
    grid[6].set_ylabel('Semantic', fontsize=fontsize)

    return {
        'rgb_input': masked_rgb,
        'rgb_pred': pred_rgb2,
        'rgb_gt': denormalize(input_dict['rgb'])[0].permute(1,2,0).detach().cpu(),
        'depth_input': masked_depth,
        'depth_pred': pred_depth2,
        'depth_gt': input_dict['depth'][0,0].detach().cpu(),
        'semseg_input': plot_semseg_gt_masked(input_dict, masks['semseg'], mask_value=1.0) if use_semseg else None,
        'semseg_pred': plot_semseg_pred_masked(input_dict['rgb'], preds['semseg'], input_dict['semseg'], masks['semseg']) if use_semseg else None,
        'semseg_gt': plot_semseg_gt(input_dict) if use_semseg else None
    }