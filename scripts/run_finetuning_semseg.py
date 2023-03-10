# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on timm, DeiT, DINO, MoCo-v3, BEiT, MAE-priv, MAE and MMSegmentation code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# https://github.com/facebookresearch/moco-v3
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/BUPT-PRIV/MAE-priv
# https://github.com/facebookresearch/mae
# https://github.com/open-mmlab/mmsegmentation
# --------------------------------------------------------

import os
import time
import json
import random
import socket
import warnings
import argparse
import datetime

from functools import partial
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import yaml

import multimae.utils as utils
import multimae.utils.data_constants as data_constants
from multimae.models import multimae
from multimae.models.input_adapters import PatchedInputAdapter, SemSegInputAdapter
from multimae.models.output_adapters import (ConvNeXtAdapter, DPTOutputAdapter,
                                      SegmenterMaskTransformerAdapter)
from multimae.utils import NativeScalerWithGradNormCount as NativeScaler
from multimae.utils import create_model
from multimae.utils.data_constants import COCO_SEMSEG_NUM_CLASSES
from multimae.utils.datasets_semseg import build_semseg_dataset, simple_transform
from multimae.utils.dist import collect_results_cpu
from multimae.utils.log_images import log_semseg_wandb
from multimae.utils.optim_factory import LayerDecayValueAssigner, create_optimizer
from multimae.utils.pos_embed import interpolate_pos_embed_multimae
from multimae.utils.semseg_metrics import mean_iou
from multimae.parsers.finetune_semseg import get_args
from pipelines.utils.log_utils import get_logger


DOMAIN_CONF = {
    'rgb': {
        'channels': 3,
        'stride_level': 1,
        'aug_type': 'image',
        'input_adapter': partial(PatchedInputAdapter, num_channels=3),
    },
    'depth': {
        'channels': 1,
        'stride_level': 1,
        'aug_type': 'mask',
        'input_adapter': partial(PatchedInputAdapter, num_channels=1),
    },
    'semseg': {
        'stride_level': 4,
        'aug_type': 'mask',
        'input_adapter': partial(SemSegInputAdapter, num_classes=COCO_SEMSEG_NUM_CLASSES,
                                 dim_class_emb=64, interpolate_class_emb=False,
                                 emb_padding_idx=COCO_SEMSEG_NUM_CLASSES),
    },
    'pseudo_semseg': {
        'aug_type': 'mask'
    },
    'mask_valid': {
        'stride_level': 1,
        'aug_type': 'mask',
    },
}


def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    # configure new logger
    dp_logger = get_logger("[MultiMAE]", double_print=True, res_dir=args.output_dir, exp_name=args.wandb_run_name)

    if not args.show_user_warnings:
        warnings.filterwarnings("ignore", category=UserWarning)

    args.in_domains = args.in_domains.split('-')
    args.out_domains = ['semseg']
    args.all_domains = list(set(args.in_domains) | set(args.out_domains))
    if args.use_mask_valid:
        args.all_domains.append('mask_valid')
    if 'rgb' not in args.all_domains:
        args.all_domains.append('rgb')
    args.num_classes_with_void = args.num_classes + 1 if args.seg_use_void_label else args.num_classes

    # Dataset stuff
    additional_targets = {domain: DOMAIN_CONF[domain]['aug_type'] for domain in args.all_domains}

    if args.aug_name == 'simple':
        train_transform = simple_transform(train=True, additional_targets=additional_targets, input_size=args.input_size)
        val_transform = simple_transform(train=False, additional_targets=additional_targets, input_size=args.input_size)
    else:
        raise ValueError(f"Invalid aug: {args.aug_name}")

    dataset_train = build_semseg_dataset(args, data_path=args.data_path, transform=train_transform)
    dataset_val = build_semseg_dataset(args, data_path=args.eval_data_path, transform=val_transform, max_images=args.max_val_images)
    
    if args.test_data_path is not None:
        dataset_test = build_semseg_dataset(args, data_path=args.test_data_path, transform=val_transform)
    else:
        dataset_test = None

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
        
    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, drop_last=True,
        )
        dp_logger.info("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                dp_logger.info('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            if dataset_test is not None:
                sampler_test = torch.utils.data.DistributedSampler(
                    dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            if dataset_test is not None:
                sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        if dataset_test is not None:
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if global_rank == 0 and args.log_wandb:
        log_writer = utils.WandbLogger(args)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None

    if dataset_test is not None:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, 
            sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_test = None

    # use psuedo labels as ground truth for training
    if 'pseudo_semseg' in args.in_domains:
        args.in_domains.remove('pseudo_semseg')
        args.in_domains.append('semseg')

    input_adapters = {
        domain: DOMAIN_CONF[domain]['input_adapter'](
            stride_level=DOMAIN_CONF[domain]['stride_level'],
            patch_size_full=args.patch_size,
            image_size=args.input_size,
            learnable_pos_emb=args.learnable_pos_emb,
        )
        for domain in args.in_domains
    }

    # DPT settings are fixed for ViT-B. Modify them if using a different backbone.
    if args.model != 'multivit_base' and args.output_adapter == 'dpt':
        raise NotImplementedError('Unsupported backbone: DPT head is fixed for ViT-B.')

    adapters_dict = {
        'segmenter': partial(SegmenterMaskTransformerAdapter, depth=args.decoder_depth, drop_path_rate=args.drop_path_decoder),
        'convnext': partial(ConvNeXtAdapter, 
                            preds_per_patch=args.decoder_preds_per_patch, depth=args.decoder_depth,
                            interpolate_mode=args.decoder_interpolate_mode, main_tasks=args.decoder_main_tasks.split('-')),
        'dpt': partial(DPTOutputAdapter, stride_level=1, main_tasks=args.decoder_main_tasks.split('-'), head_type='semseg'),
    }

    output_adapters = {
        'semseg': adapters_dict[args.output_adapter](
            num_classes=args.num_classes_with_void,
            embed_dim=args.decoder_dim, 
            patch_size=args.patch_size,
        ),
    }

    model = create_model(
        args.model,
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        drop_path_rate=args.drop_path_encoder,
    )

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu')
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']

        class_emb_key = 'input_adapters.semseg.class_emb.weight'
        if class_emb_key in checkpoint_model:
            checkpoint_model[class_emb_key] = F.pad(checkpoint_model[class_emb_key], (0, 0, 0, 1))

        # Remove output adapters
        for k in list(checkpoint_model.keys()):
            if "output_adapters" in k:
                del checkpoint_model[k]

        # Interpolate position embedding
        interpolate_pos_embed_multimae(model, checkpoint_model)

        # Load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    dp_logger.info("Model = %s" % str(model_without_ddp))
    dp_logger.info('number of params: {} M'.format(n_parameters / 1e6))

    total_batch_size = args.batch_size * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size

    dp_logger.info("LR = %.8f" % args.lr)
    dp_logger.info("Batch size = %d" % total_batch_size)
    dp_logger.info("Number of training steps = %d" % num_training_steps_per_epoch)
    dp_logger.info("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        dp_logger.info("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    dp_logger.info("Skip weight decay list: ", skip_weight_decay_list)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module

    optimizer = create_optimizer(args, model_without_ddp, skip_list=skip_weight_decay_list,
                                 get_num_layer=assigner.get_layer_id if assigner is not None else None,
                                 get_layer_scale=assigner.get_scale if assigner is not None else None)
    loss_scaler = NativeScaler(enabled=args.fp16)

    dp_logger.info("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    dp_logger.info("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    criterion = torch.nn.CrossEntropyLoss(ignore_index=utils.SEG_IGNORE_INDEX)

    dp_logger.info("criterion = %s" % str(criterion))

    # Specifies if transformer encoder should only return last layer or all layers for DPT
    return_all_layers = args.output_adapter in ['dpt']

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        val_stats = evaluate(model=model, criterion=criterion, data_loader=data_loader_val,
                             device=device, epoch=-1, in_domains=args.in_domains,
                             num_classes=args.num_classes, dataset_name=args.dataset_name, mode='val',
                             fp16=args.fp16, return_all_layers=return_all_layers, dp_logger=dp_logger)
        dp_logger.info(f"Performance of the network on the {len(dataset_val)} validation images")
        miou, a_acc, acc, loss = val_stats['mean_iou'], val_stats['pixel_accuracy'], val_stats['mean_accuracy'], val_stats['loss']
        dp_logger.info(f'* mIoU {miou:.3f} aAcc {a_acc:.3f} Acc {acc:.3f} Loss {loss:.3f}')
        exit(0)

    if args.test:
        test_stats = evaluate(model=model, criterion=criterion, data_loader=data_loader_test,
                              device=device, epoch=-1, in_domains=args.in_domains,
                              num_classes=args.num_classes, dataset_name=args.dataset_name, mode='test',
                              fp16=args.fp16, return_all_layers=return_all_layers, dp_logger=dp_logger)
        dp_logger.info(f"Performance of the network on the {len(dataset_test)} test images")
        miou, a_acc, acc, loss = test_stats['mean_iou'], test_stats['pixel_accuracy'], test_stats['mean_accuracy'], test_stats['loss']
        dp_logger.info(f'* mIoU {miou:.3f} aAcc {a_acc:.3f} Acc {acc:.3f} Loss {loss:.3f}')
        exit(0)

    dp_logger.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_miou = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        train_stats = train_one_epoch(
            model=model, criterion=criterion, data_loader=data_loader_train,
            optimizer=optimizer, device=device, epoch=epoch, loss_scaler=loss_scaler,
            max_norm=args.clip_grad, log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values, in_domains=args.in_domains,
            fp16=args.fp16, return_all_layers=return_all_layers, dp_logger=dp_logger
        )
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)
                dp_logger.info("save model at epoch %d" % epoch)

        if data_loader_val is not None and (epoch % args.eval_freq == 0 or epoch == args.epochs - 1):
            log_images = args.log_wandb and args.log_images_wandb and (epoch % args.log_images_freq == 0)
            val_stats = evaluate(model=model, criterion=criterion, data_loader=data_loader_val,
                                 device=device, epoch=epoch, in_domains=args.in_domains,
                                 num_classes=args.num_classes, log_images=log_images, 
                                 dataset_name=args.dataset_name, mode='val', fp16=args.fp16,
                                 return_all_layers=return_all_layers, dp_logger=dp_logger)
            if max_miou < val_stats["mean_iou"]:
                max_miou = val_stats["mean_iou"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best")
                    dp_logger.info("save best model at epoch %d" % epoch)
                    
            dp_logger.info(f'Max mIoU: {max_miou:.3f}')
            log_stats = {**{f'train/{k}': v for k, v in train_stats.items()},
                         **{f'val/{k}': v for k, v in val_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train/{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if log_writer is not None:
            log_writer.update(log_stats)

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    dp_logger.info('Training time {}'.format(total_time_str))

    # Test with best checkpoint
    if data_loader_test is not None:
        dp_logger.info('Loading model with best validation mIoU')
        checkpoint = torch.load(os.path.join(args.output_dir, 'checkpoint-best.pth'), map_location='cpu')
        state_dict = {}
        for k,v in checkpoint['model'].items():
            state_dict[f'module.{k}'] = v
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)

        dp_logger.info('Testing with best checkpoint')
        test_stats = evaluate(model=model, criterion=criterion, data_loader=data_loader_test,
                              device=device, epoch=checkpoint['epoch'], in_domains=args.in_domains,
                              num_classes=args.num_classes, log_images=True, dataset_name=args.dataset_name,
                              mode='test', fp16=args.fp16, return_all_layers=return_all_layers, dp_logger=dp_logger)
        log_stats = {f'test/{k}': v for k, v in test_stats.items()}
        if log_writer is not None:
            log_writer.set_step(args.epochs * num_training_steps_per_epoch)
            log_writer.update(log_stats)
        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device, epoch: int,
                    loss_scaler, max_norm: float = 0, log_writer=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, in_domains=None, fp16=True,
                    return_all_layers=False, dp_logger=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    for step, (x, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header, dp_logger)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        tasks_dict = {
            task: tensor.to(device, non_blocking=True)
            for task, tensor in x.items()
        }

        input_dict = {
            task: tensor
            for task, tensor in tasks_dict.items()
            if task in in_domains
        }

        if 'pseudo_semseg' in tasks_dict and 'semseg' in in_domains:
            psemseg  = tasks_dict['pseudo_semseg']
            psemseg[psemseg > COCO_SEMSEG_NUM_CLASSES - 1] = COCO_SEMSEG_NUM_CLASSES
            input_dict['semseg'] = psemseg
        
        # Forward + backward
        with torch.cuda.amp.autocast(enabled=fp16):
            preds = model(input_dict, return_all_layers=return_all_layers)
            seg_pred, seg_gt = preds['semseg'], tasks_dict['semseg']
            loss = criterion(seg_pred, seg_gt)

        loss_value = loss.item()

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        if fp16:
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        # Metrics and logging
        metric_logger.update(loss=loss_value)
        if fp16:
            metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(
                {
                    'loss': loss_value,
                    'lr': max_lr,
                    'weight_decay': weight_decay_value,
                    'grad_norm': grad_norm,
                }
            )
            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if dp_logger is not None:
        dp_logger.info("Averaged stats:", metric_logger)
    else:
        print("Averaged stats:", metric_logger)
    return {'[Epoch] ' + k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, epoch, in_domains, num_classes, dataset_name,
             log_images=False, mode='val', fp16=True, return_all_layers=False, dp_logger=None):
    # Switch to evaluation mode
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if mode == 'val':
        header = '(Eval) Epoch: [{}]'.format(epoch)
    elif mode == 'test':
        header = '(Test) Epoch: [{}]'.format(epoch)
    else:
        raise ValueError(f'Invalid eval mode {mode}')
    print_freq = 20

    seg_preds = []
    seg_gts = []

    rgb_gts = None
    seg_preds_with_void = None
    if log_images:
        rgb_gts = []
        seg_preds_with_void = []
        depth_gts = []
        
    for (x, _) in metric_logger.log_every(data_loader, print_freq, header, dp_logger):
        tasks_dict = {
            task: tensor.to(device, non_blocking=True)
            for task, tensor in x.items()
        }

        input_dict = {
            task: tensor
            for task, tensor in tasks_dict.items()
            if task in in_domains
        }

        if 'pseudo_semseg' in tasks_dict and 'semseg' in in_domains:
            psemseg  = tasks_dict['pseudo_semseg']
            psemseg[psemseg == 254] = COCO_SEMSEG_NUM_CLASSES
            input_dict['semseg'] = psemseg

        # Forward + backward
        with torch.cuda.amp.autocast(enabled=fp16):
            preds = model(input_dict, return_all_layers=return_all_layers)
            seg_pred, seg_gt = preds['semseg'], tasks_dict['semseg']
            loss = criterion(seg_pred, seg_gt)

        loss_value = loss.item()
        # If there is void, exclude it from the preds and take second highest class
        seg_pred_argmax = seg_pred[:, :num_classes].argmax(dim=1)
        seg_preds.extend(list(seg_pred_argmax.cpu().numpy()))
        seg_gts.extend(list(seg_gt.cpu().numpy()))

        if log_images:
            rgb_gts.extend(tasks_dict['rgb'].cpu().unbind(0))
            seg_preds_with_void.extend(list(seg_pred.argmax(dim=1).cpu().numpy()))
            if 'depth' in tasks_dict:
                depth_gts.extend(tasks_dict['depth'].cpu().unbind(0))

        metric_logger.update(loss=loss_value)

    # Do before metrics so that void is not replaced
    if log_images and utils.is_main_process():
        prefix = 'val/img' if mode == 'val' else 'test/img'
        log_semseg_wandb(rgb_gts, seg_preds_with_void, seg_gts, depth_gts=depth_gts, dataset_name=dataset_name, prefix=prefix)

    scores = compute_metrics_distributed(seg_preds, seg_gts, size=len(data_loader.dataset), num_classes=num_classes,
                                         device=device, ignore_index=utils.SEG_IGNORE_INDEX, dist_on=None)

    for k, v in scores.items():
        metric_logger.update(**{f"{k}": v})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    dp_logger.info(f'* mIoU {metric_logger.mean_iou.global_avg:.3f} aAcc {metric_logger.pixel_accuracy.global_avg:.3f} '
          f'Acc {metric_logger.mean_accuracy.global_avg:.3f} Loss {metric_logger.loss.global_avg:.3f}')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    
def compute_metrics_distributed(seg_preds, seg_gts, size, num_classes, device, ignore_index=utils.SEG_IGNORE_INDEX, dist_on='cpu'):

    # Replace void by ignore in gt (void is never counted in mIoU)
    for seg_gt in seg_gts:
        # Void label is equal to num_classes
        seg_gt[seg_gt == num_classes] = ignore_index

    # Collect metrics from all devices
    if dist_on == 'cpu':
        all_seg_preds = collect_results_cpu(seg_preds, size, tmpdir=None)
        all_seg_gts = collect_results_cpu(seg_gts, size, tmpdir=None)
    elif dist_on == 'gpu':
        world_size = utils.get_world_size()
        all_seg_preds = [None for _ in range(world_size)]
        all_seg_gts = [None for _ in range(world_size)]
        # gather all result part
        dist.all_gather_object(all_seg_preds, seg_preds)
        dist.all_gather_object(all_seg_gts, seg_gts)
    else:
        # not using distribute mode
        all_seg_preds = [seg_preds]
        all_seg_gts = [seg_gts]

    ret_metrics_mean = torch.zeros(3, dtype=float, device=device)

    if utils.is_main_process():
        ordered_seg_preds = [result for result_part in all_seg_preds for result in result_part]
        ordered_seg_gts = [result for result_part in all_seg_gts for result in result_part]

        ret_metrics = mean_iou(results=ordered_seg_preds,
                               gt_seg_maps=ordered_seg_gts,
                               num_classes=num_classes,
                               ignore_index=ignore_index)

        ret_metrics_mean = torch.tensor(
            [
                np.round(np.nanmean(ret_metric.astype(float)) * 100, 2)
                for ret_metric in ret_metrics
            ],
            dtype=float,
            device=device,
        )
        # cat_iou = ret_metrics[2]

    if dist_on is not None:
        # broadcast metrics from 0 to all nodes
        dist.broadcast(ret_metrics_mean, 0)
        
    pix_acc, mean_acc, miou = ret_metrics_mean
    ret = dict(pixel_accuracy=pix_acc, mean_accuracy=mean_acc, mean_iou=miou)
    return ret


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
