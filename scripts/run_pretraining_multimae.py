# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on timm, DeiT, DINO, MoCo-v3, BEiT, MAE-priv and MAE code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# https://github.com/facebookresearch/moco-v3
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/BUPT-PRIV/MAE-priv
# https://github.com/facebookresearch/mae
# --------------------------------------------------------

import json
import math
import os
import sys
import time
import random
import warnings
import datetime
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

# do not forget to import `multimae` here
from multimae.models import multimae
import multimae.utils as utils
import multimae.utils.data_constants as data_constants
from multimae.utils.train_utils import normalize_depth
from multimae.models.criterion import (MaskedCrossEntropyLoss, MaskedL1Loss,
                                MaskedMSELoss)
from multimae.models.input_adapters import PatchedInputAdapter, SemSegInputAdapter
from multimae.models.output_adapters import SpatialOutputAdapter
from multimae.utils import NativeScalerWithGradNormCount as NativeScaler
from multimae.utils.model_builder import create_model
from multimae.utils.data_constants import CUSTOM_SEMSEG_NUM_CLASSES, COCO_SEMSEG_NUM_CLASSES
from multimae.utils.datasets import build_multimae_pretraining_dataset
from multimae.utils.optim_factory import create_optimizer
from multimae.parsers.pretrain_multimae import get_args
from multimae.utils.log_images import log_multimae_semseg_wandb
from multimae.utils.plot_utils import get_semseg_metadata
from multimae.utils.task_balancing import (NoWeightingStrategy,
                                  UncertaintyWeightingStrategy)
from pipelines.utils.log_utils import get_logger


DOMAIN_CONF = {
    'rgb': {
        'channels': 3,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=3),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=3),
        'loss': MaskedMSELoss,
    },
    'depth': {
        'channels': 1,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=1),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=1),
        'loss': MaskedL1Loss,
    },
    'semseg': {
        'num_classes': COCO_SEMSEG_NUM_CLASSES,
        'stride_level': 1,
        'input_adapter': partial(SemSegInputAdapter, num_classes=COCO_SEMSEG_NUM_CLASSES,
                                 dim_class_emb=64, interpolate_class_emb=False),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=COCO_SEMSEG_NUM_CLASSES),
        'loss': partial(MaskedCrossEntropyLoss, label_smoothing=0.0),
    },
}


def get_model(args):
    """Creates and returns model from arguments
    """
    print(f"Creating model: {args.model} for inputs {args.in_domains} and outputs {args.out_domains}")

    input_adapters = {
        domain: DOMAIN_CONF[domain]['input_adapter'](
            stride_level=DOMAIN_CONF[domain]['stride_level'],
            patch_size_full=args.patch_size,
        )
        for domain in args.in_domains
    }

    output_adapters = {
        domain: DOMAIN_CONF[domain]['output_adapter'](
            stride_level=DOMAIN_CONF[domain]['stride_level'],
            patch_size_full=args.patch_size,
            dim_tokens=args.decoder_dim,
            depth=args.decoder_depth,
            num_heads=args.decoder_num_heads,
            use_task_queries=args.decoder_use_task_queries,
            task=domain,
            context_tasks=list(args.in_domains),
            use_xattn=args.decoder_use_xattn
        )
        for domain in args.out_domains
    }

    # Add normalized pixel output adapter if specified
    if args.extra_norm_pix_loss:
        output_adapters['norm_rgb'] = DOMAIN_CONF['rgb']['output_adapter'](
            stride_level=DOMAIN_CONF['rgb']['stride_level'],
            patch_size_full=args.patch_size,
            dim_tokens=args.decoder_dim,
            depth=args.decoder_depth,
            num_heads=args.decoder_num_heads,
            use_task_queries=args.decoder_use_task_queries,
            task='rgb',
            context_tasks=list(args.in_domains),
            use_xattn=args.decoder_use_xattn
        )
    
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        num_global_tokens=args.num_global_tokens,
        drop_path_rate=args.drop_path
    ) 

    return model

def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # Fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    # configure new logger
    dp_logger = get_logger("[MultiMAE]", 
                           double_print=True, 
                           res_dir=args.output_dir, 
                           exp_name=args.wandb_run_name)

    if not args.show_user_warnings:
        warnings.filterwarnings("ignore", category=UserWarning)

    # reconfigure args
    args.in_domains = args.in_domains.split('-')
    args.out_domains = args.out_domains.split('-')
    args.all_domains = list(set(args.in_domains) | set(args.out_domains))
    DOMAIN_CONF['semseg']['stride_level'] = args.semseg_stride_level

    model = get_model(args)

    if args.task_balancer == 'uncertainty':
        loss_balancer = UncertaintyWeightingStrategy(tasks=args.out_domains)
    else:
        loss_balancer = NoWeightingStrategy()

    tasks_loss_fn = {
        domain: DOMAIN_CONF[domain]['loss'](patch_size=args.patch_size, 
                                            stride=DOMAIN_CONF[domain]['stride_level'])
        for domain in args.out_domains
    }

    # Add normalized pixel loss if specified
    if args.extra_norm_pix_loss:
        tasks_loss_fn['norm_rgb'] = DOMAIN_CONF['rgb']['loss'](patch_size=args.patch_size,
                                                               stride=DOMAIN_CONF['rgb']['stride_level'],
                                                               norm_pix=True)

    # Get dataset
    dataset_train = build_multimae_pretraining_dataset(args, args.data_path)
    use_validation = True if args.eval_data_path else False
    dataset_val = build_multimae_pretraining_dataset(args, args.eval_data_path) if use_validation else None
    
    num_tasks = utils.get_world_size()  # return 1 for non-distributed device
    global_rank = utils.get_rank()      # return 0 for non-distributed device
    num_training_steps_per_epoch = len(dataset_train) // args.batch_size // num_tasks
        
    if args.distributed:
        sampler_rank = global_rank
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True, drop_last=True,
        )
        dp_logger.info("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.RandomSampler(dataset_val) if use_validation else None

    if global_rank == 0 and args.log_wandb:
        log_writer = utils.WandbLogger(args)
    else:
        log_writer = None

    print(vars(args))
    
    # configure for detectron dataset (for prediection)
    metadata = get_semseg_metadata(args.eval_data_path)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    ) if use_validation else None

    model.to(device)
    loss_balancer.to(device)
    model_without_ddp = model
    loss_balancer_without_ddp = loss_balancer
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    dp_logger.info(f"Model = %s" % str(model_without_ddp))
    dp_logger.info(f"Number of params: {n_parameters / 1e6} M")

    total_batch_size = args.batch_size * utils.get_world_size()
    args.lr = args.blr * total_batch_size / 256

    dp_logger.info("LR = %.8f" % args.lr)
    dp_logger.info("Batch size = %d" % total_batch_size)
    dp_logger.info("Number of training steps = %d" % num_training_steps_per_epoch)
    dp_logger.info("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.distributed and args.task_balancer != 'none':
        loss_balancer = torch.nn.parallel.DistributedDataParallel(loss_balancer, device_ids=[args.gpu])
        loss_balancer_without_ddp = loss_balancer.module

    optimizer = create_optimizer(
        args, {'model': model_without_ddp, 'balancer': loss_balancer_without_ddp})
    loss_scaler = NativeScaler()

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

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    dp_logger.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        train_stats = train_one_epoch(
            model=model,
            data_loader=data_loader_train,
            tasks_loss_fn=tasks_loss_fn,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            loss_balancer=loss_balancer,
            loss_scaler=loss_scaler,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            args=args,
            
            log_writer=log_writer,
            dp_logger=dp_logger
        )
        
        log_stats = deepcopy(train_stats)
        
        if use_validation and (epoch % args.eval_freq == 0 or epoch == args.epochs - 1):
            val_stats = validate(
                model=model,
                data_loader=data_loader_val,
                tasks_loss_fn=tasks_loss_fn,
                loss_balancer=loss_balancer,
                device=device,
                epoch=epoch,
                metadata=metadata,
                args=args,
                dp_logger=dp_logger
            )
            log_stats.update(val_stats)
        
        if log_writer is not None:
            log_writer.update({**{k: v for k, v in log_stats.items()}, 'epoch': epoch})
            
        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, loss_balancer=loss_balancer_without_ddp, epoch=epoch)

        log_stats.update({'epoch': epoch, 'n_parameters': n_parameters})

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    dp_logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, tasks_loss_fn: Dict[str, torch.nn.Module],
                    loss_balancer: torch.nn.Module, optimizer: torch.optim.Optimizer, start_steps: int, 
                    device: torch.device, epoch: int, loss_scaler, args, 
                    lr_schedule_values: List[float] = None, wd_schedule_values: List[float] = None, 
                    log_writer=None, lr_scheduler=None, dp_logger=None):    
    # assign args values to local variables
    max_norm=args.clip_grad
    max_skip_norm=args.skip_grad
    num_encoded_tokens=args.num_encoded_tokens
    in_domains=args.in_domains
    loss_on_unmasked=args.loss_on_unmasked
    alphas=args.alphas
    sample_tasks_uniformly=args.sample_tasks_uniformly
    standardize_depth=args.standardize_depth
    extra_norm_pix_loss=args.extra_norm_pix_loss
    fp32_output_adapters=args.fp32_output_adapters.split('-')
    mask_type=args.mask_type
    masked_rgb_gate_only=args.masked_rgb_gate_only
    
    if loss_on_unmasked:
        task_loss_types = {}
        tasks_loss_unmask = loss_on_unmasked.split('-')
        for task in tasks_loss_fn:
            task_loss_types[task] = 'unmask' if task in tasks_loss_unmask else 'mask'
    else:
        task_loss_types = {task: 'mask' for task in tasks_loss_fn.keys()}
        
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
        
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

        # Truncated depth standardization
        if standardize_depth and 'depth' in tasks_dict:
            tasks_dict["depth"] = normalize_depth(tasks_dict["depth"])

        input_dict = {
            task: tensor
            for task, tensor in tasks_dict.items()
            if task in in_domains
        }

        with torch.cuda.amp.autocast():
            preds, masks = model(
                input_dict, 
                num_encoded_tokens=num_encoded_tokens, 
                alphas=alphas, 
                sample_tasks_uniformly=sample_tasks_uniformly,
                fp32_output_adapters=fp32_output_adapters,
                mask_type=mask_type,
                masked_rgb_gate_only=masked_rgb_gate_only,
                semseg_gt=tasks_dict["semseg"],
                in_domains=in_domains,
                semseg_stride=args.semseg_stride_level
            )

            if extra_norm_pix_loss:
                tasks_dict['norm_rgb'] = tasks_dict['rgb']
                masks['norm_rgb'] = masks.get('rgb', None)

            task_losses = {}
    
            for task in preds:
                target = tasks_dict[task]
                unmask_loss = task_loss_types[task] == 'unmask'
                if unmask_loss:
                    task_losses[task] = tasks_loss_fn[task](preds[task].float(), target)
                else:
                    task_losses[task] = tasks_loss_fn[task](preds[task].float(), target, mask=masks.get(task, None))

            weighted_task_losses = loss_balancer(task_losses)
            loss = sum(weighted_task_losses.values())

        loss_value = sum(task_losses.values()).item()
        task_loss_values = {f'{task}_loss': l.item() for task, l in task_losses.items()}
        weighted_task_loss_values = {f'{task}_loss_weighted': l.item() for task, l in weighted_task_losses.items()}

        if not math.isfinite(loss_value):
            dp_logger.info("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm, skip_grad=max_skip_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(**task_loss_values)
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
                    'train/loss': loss_value,
                    'train/lr': max_lr,
                    'train/weight_decay': weight_decay_value,
                    'train/grad_norm': grad_norm,
                }
            )
            log_task_loss_values = {"train/"+k: v for k, v in task_loss_values.items()}
            log_weighted_task_loss_values = {"train/"+k: v for k, v in weighted_task_loss_values.items()}
            log_writer.update(log_task_loss_values)
            log_writer.update(log_weighted_task_loss_values)
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if dp_logger is not None:
        dp_logger.info("Averaged stats:", metric_logger)
    else:
        print("Averaged stats:", metric_logger)
    return {'train/avg_' + k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def validate(model: torch.nn.Module, data_loader: Iterable, tasks_loss_fn: Dict[str, torch.nn.Module],
            loss_balancer: torch.nn.Module, device: torch.device, epoch: int, args, metadata=None, dp_logger=None):
    
    # assgin parser args to local variables
    num_encoded_tokens=args.num_encoded_tokens
    in_domains=args.in_domains
    loss_on_unmasked=args.loss_on_unmasked
    alphas=args.alphas
    sample_tasks_uniformly=args.sample_tasks_uniformly
    standardize_depth=args.standardize_depth
    extra_norm_pix_loss=args.extra_norm_pix_loss
    fp32_output_adapters=args.fp32_output_adapters.split('-')
    mask_type=args.mask_type
    log_images=args.wandb_log_img
    masked_rgb_gate_only=args.masked_rgb_gate_only
    
    if loss_on_unmasked:
        task_loss_types = {}
        tasks_loss_unmask = loss_on_unmasked.split('-')
        for task in tasks_loss_fn.keys():
            if task in tasks_loss_unmask:
                task_loss_types[task] = 'unmask'
            else:
                task_loss_types[task] = 'mask'
    else:
        task_loss_types = {task: 'mask' for task in tasks_loss_fn.keys()}
    
    if log_images:
        log_inputs, log_preds,log_masks = None, None, None
        
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = '(Val) Epoch: [{}]'.format(epoch)
    print_freq = 10

    for step, (x, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header, dp_logger)):
        tasks_dict = {
            task: tensor.to(device, non_blocking=True)
            for task, tensor in x.items()
        }

        # Truncated depth standardization
        if standardize_depth and 'depth' in tasks_dict:
            tasks_dict["depth"] = normalize_depth(tasks_dict["depth"])

        input_dict = {
            task: tensor
            for task, tensor in tasks_dict.items()
            if task in in_domains
        }

        preds, masks = model(
            input_dict, 
            num_encoded_tokens=num_encoded_tokens, 
            alphas=alphas, 
            sample_tasks_uniformly=sample_tasks_uniformly,
            fp32_output_adapters=fp32_output_adapters,
            mask_type=mask_type,
            masked_rgb_gate_only=masked_rgb_gate_only,
            semseg_gt=tasks_dict["semseg"],
            in_domains=in_domains,
            semseg_stride=args.semseg_stride_level
        )

        if extra_norm_pix_loss:
            tasks_dict['norm_rgb'] = tasks_dict['rgb']
            masks['norm_rgb'] = masks.get('rgb', None)

        task_losses = {}
        
        for task in preds:
            target = tasks_dict[task]
            unmask_loss = task_loss_types[task] == 'unmask'
                
            if unmask_loss:
                task_losses[task] = tasks_loss_fn[task](preds[task].float(), target)
            else:
                task_losses[task] = tasks_loss_fn[task](preds[task].float(), target, mask=masks.get(task, None))

        weighted_task_losses = loss_balancer(task_losses)

        loss_value = sum(task_losses.values()).item()
        task_loss_values = {f'{task}_loss': l.item() for task, l in task_losses.items()}
        weighted_task_loss_values = {f'{task}_loss_weighted': l.item() for task, l in weighted_task_losses.items()}

        if not math.isfinite(loss_value):
            dp_logger.info("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
            
        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(**task_loss_values)
        metric_logger.update(**weighted_task_loss_values)
        
        if log_images and log_inputs is None:
            log_inputs, log_preds, log_masks = tasks_dict, preds, masks
        
    eval_metrics = {"val/" + k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    if log_images and utils.is_main_process():
        log_multimae_semseg_wandb(log_inputs, log_preds, log_masks, prefix='plots/val', 
                                  metadata=metadata, semseg_stride=args.semseg_stride_level)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if dp_logger is not None:
        dp_logger.info("Averaged stats:", metric_logger)
    else:
        print("Averaged stats:", metric_logger)
    return eval_metrics


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
