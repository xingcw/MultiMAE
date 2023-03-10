import os
import yaml
import datetime
import argparse
from pathlib import Path
from multimae.utils.data_constants import ADE_TRAIN_PATH, ADE_VAL_PATH


def get_args():
    config_parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    config_parser.add_argument('-c', '--config', default='cfgs/finetune/semseg/nyu/ft_nyu_200e_multimae-b_rgb-depth.yaml', 
                               type=str, metavar='FILE', help='YAML config file specifying default arguments')

    parser = argparse.ArgumentParser('MultiMAE semantic segmentation fine-tuning script', add_help=False)
    
    # configs from file
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')
    
    # training parameters
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=64, type=int)
    parser.add_argument('--save_ckpt_freq', default=20, type=int)

    # Task parameters
    parser.add_argument('--in_domains', default='rgb', type=str,
                        help='Input domain names, separated by hyphen')
    parser.add_argument('--standardize_depth', action='store_true')
    parser.add_argument('--no_standardize_depth', action='store_false', dest='standardize_depth')
    parser.set_defaults(standardize_depth=True)
    parser.add_argument('--use_mask_valid', action='store_true')
    parser.add_argument('--no_mask_valid', action='store_false', dest='use_mask_valid')
    parser.set_defaults(use_mask_valid=False)
    parser.add_argument('--load_pseudo_depth', action='store_true')
    parser.add_argument('--no_load_pseudo_depth', action='store_false', dest='load_pseudo_depth')
    parser.set_defaults(load_pseudo_depth=False)

    # Model parameters
    parser.add_argument('--model', default='multivit_base', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--num_global_tokens', default=1, type=int,
                        help='number of global tokens to add to encoder')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='base patch size for image-like modalities')
    parser.add_argument('--input_size', default=512, type=int,
                        help='images input size for backbone')
    parser.add_argument('--drop_path_encoder', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--learnable_pos_emb', action='store_true',
                        help='Makes the positional embedding learnable')
    parser.add_argument('--no_learnable_pos_emb', action='store_false', dest='learnable_pos_emb')
    parser.set_defaults(learnable_pos_emb=False)

    parser.add_argument('--output_adapter', type=str, default='convnext',
                        choices=['segmenter', 'convnext', 'dpt'],
                        help='One of [segmenter,  convnext, dpt] (default: convnext)')
    parser.add_argument('--decoder_dim', default=6144, type=int,
                        help='Token dimension for the decoder layers, for convnext and segmenter adapters')
    parser.add_argument('--decoder_depth', default=4, type=int,
                        help='Depth of decoder (for convnext and segmenter adapters')
    parser.add_argument('--drop_path_decoder', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--decoder_preds_per_patch', type=int, default=16,
                        help='Predictions per patch for convnext adapter')
    parser.add_argument('--decoder_interpolate_mode', type=str, default='bilinear',
                        choices=['bilinear', 'nearest'], help='for convnext adapter')
    parser.add_argument('--decoder_main_tasks', type=str, default='rgb',
                        help='for convnext adapter, separate tasks with a hyphen')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=[0.9, 0.999], type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")
    parser.add_argument('--decoder_decay', type=float, default=None,
                        help='decoder weight decay')
    parser.add_argument('--no_lr_scale_list', type=str, default='',
                        help='Weights that should not be affected by layer decay rate, separated by hyphen.')

    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=0.0, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (0.0)')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA')

    parser.add_argument('--warmup_epochs', type=int, default=1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    # Augmentation parameters
    parser.add_argument('--aug_name', type=str, default='simple',
                        choices=['simple'],
                        help='One of [simple] (default: simple)')

    # Finetuning parameters
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--num_classes', default=150, type=str, help='number of semantic classes')
    parser.add_argument('--dataset_name', default='ade20k', type=str, help='dataset name for plotting')
    parser.add_argument('--data_path', default=ADE_TRAIN_PATH, type=str, help='dataset path')
    parser.add_argument('--eval_data_path', default=ADE_VAL_PATH, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--test_data_path', default=None, type=str,
                        help='dataset path for testing')
    parser.add_argument('--max_val_images', default=None, type=int,
                        help='maximum number of validation images. (default: None)')
    parser.add_argument('--eval_freq', default=1, type=int, help="frequency of evaluation")
    parser.add_argument('--seg_reduce_zero_label', action='store_true',
                        help='set label 0 to ignore, reduce all other labels by 1')
    parser.add_argument('--seg_use_void_label', action='store_true', help='label border as void instead of ignore')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--test', action='store_true',
                        help='Perform testing only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                    help='Enabling distributed evaluation')
    parser.add_argument('--no_dist_eval', action='store_false', dest='dist_eval',
                    help='Disabling distributed evaluation')
    parser.set_defaults(dist_eval=False)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--no_find_unused_params', action='store_false', dest='find_unused_params')
    parser.set_defaults(find_unused_params=True)

    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--no_fp16', action='store_false', dest='fp16')
    parser.set_defaults(fp16=True)

    # Wandb logging
    parser.add_argument('--redirect', "-rd", default="", type=str,
                        help="change the result dir to redirected folder")
    parser.add_argument('--log_wandb', default=False, action='store_true',
                        help='log training and validation metrics to wandb')
    parser.add_argument('--wandb_project', default=None, type=str,
                        help='log training and validation metrics to wandb')
    parser.add_argument('--wandb_entity', default=None, type=str,
                        help='user or team name of wandb')
    parser.add_argument('--wandb_run_name', default=None, type=str,
                        help='run name on wandb')
    parser.add_argument('--log_images_wandb', action='store_true')
    parser.add_argument('--log_images_freq', default=5, type=int,
                        help="Frequency of image logging (in epochs)")
    parser.add_argument('--show_user_warnings', default=False, action='store_true')

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Do we have a config file to parse?
    multimae_path = Path(os.environ["FLIGHTMARE_PATH"]).parent / "vision_backbones/MultiMAE"
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        config_path = multimae_path / args_config.config
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)
    
    # add prefix for all paths
    args.data_path = str(multimae_path / args.data_path)
    args.eval_data_path = str(multimae_path / args.eval_data_path)
    args.finetune = str(multimae_path / args.finetune)
    
    # configure save dir
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    args.output_dir = str(multimae_path / args.output_dir / timestamp)

    return args