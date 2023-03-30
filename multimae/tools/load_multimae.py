import os
import yaml
import socket
from pathlib import Path

import torch
from multimae.models.criterion import MaskedCrossEntropyLoss, MaskedMSELoss, MaskedL1Loss

from functools import partial
from multimae.models.input_adapters import PatchedInputAdapter, SemSegInputAdapter
from multimae.models.output_adapters import SpatialOutputAdapter
from multimae.utils.model_builder import create_model
from multimae.utils.train_utils import normalize_depth
from multimae.parsers.pretrain_multimae import get_args

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


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
        'num_classes': 133,
        'stride_level': 4,
        'input_adapter': partial(SemSegInputAdapter, num_classes=133,
                                 dim_class_emb=64, interpolate_class_emb=False),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=133),
        'loss': partial(MaskedCrossEntropyLoss, label_smoothing=0.0),
    },
}


MODELS = {
        "unmask-semseg-loss": "03-11-17-16-39/checkpoint-299.pth",
        "gate-half-depth-semseg": "03-11-23-14-17/checkpoint-299.pth",
        "no-standard-depth": "03-12-11-46-23/checkpoint-299.pth",
        "only-gate-mask-rgb_unmask-depth-loss": "03-12-18-08-15/checkpoint-299.pth",
        "depth-semseg": "03-20-17-51-50/checkpoint-999.pth",
        "fine-semseg": "03-13-11-59-42/checkpoint-299.pth",
        "depth": "03-23-21-53-07/checkpoint-999.pth",
        "semseg": "03-23-09-07-13/checkpoint-999.pth",
        "rgb-depth": "03-23-01-05-28/checkpoint-999.pth",
        "rgb-only": "03-25-20-30-23/checkpoint-999.pth",
}

def load_model(model_name):
    
    flightmare_path = Path(os.environ["FLIGHTMARE_PATH"])
    multimae_path = flightmare_path.parent / "vision_backbones/MultiMAE"

    device = torch.device('cuda')
    server = socket.gethostname()
    if server == "snaga":
        multimae_path = Path("/data/storage/chunwei/multimae")
        
    if model_name == "pretrained":
        CKPT_URL = 'https://github.com/EPFL-VILAB/MultiMAE/releases/download/pretrained-weights/multimae-b_98_rgb+-depth-semseg_1600e_multivit-afff3f8c.pth'
        ckpt = torch.hub.load_state_dict_from_url(CKPT_URL, map_location='cpu')
        # load args from argparser
        args = get_args(no_command_line_args=True)
        # load default configs from checkpoint config file
        pretrained_config_path = multimae_path / "cfgs/pretrain/multimae-b_98_rgb+-depth-semseg_1600e.yaml"
        with open(pretrained_config_path, 'r') as f:
            pretrained_config = yaml.safe_load(f)
            f.close()
        # update args with pretrained config
        for k, v in pretrained_config.items():
            setattr(args, k, v)
        # reconfigure args
        args.in_domains = args.in_domains.split('-')
        args.out_domains = args.out_domains.split('-')
        args.all_domains = list(set(args.in_domains) | set(args.out_domains))
        print("Loaded pretrained model from: ", CKPT_URL)
        print(vars(args))
            
    elif model_name in MODELS:
        pretrained_model_path = multimae_path / f"results/pretrain/{MODELS[model_name]}"
        ckpt = torch.load(pretrained_model_path, map_location='cpu')
        args = ckpt["args"]
        print("Model loaded from: ", pretrained_model_path)
    else:
        raise KeyError(f"Model name {model_name} not found.")
    
    args.semseg_stride_level = 4
    DOMAIN_CONF['semseg']['stride_level'] = args.semseg_stride_level

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
        pretrained=False,
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        num_global_tokens=args.num_global_tokens,
        drop_path_rate=args.drop_path
    ) 
    
    model.load_state_dict(ckpt['model'], strict=True)   
    model.to(device)
    model.eval()
    
    return model, args


def multimae_predict(inputs, model, model_configs=None, return_embed=True, device="cuda", semseg_stride=4, patch_dims=(14, 14)):
    
    model_name = model_configs["checkpoint_key"]
        
    if model_name != "no-standard-depth" and "depth" in inputs:
        inputs["depth"] = normalize_depth(inputs["depth"])
        
    inputs = {k: v.to(device) for k,v in inputs.items()}
    in_domains = list(inputs.keys())
    
    patch_x, patch_y = patch_dims
    num_patchs = patch_x * patch_y
    num_encoded_tokens = num_patchs * len(in_domains)
    B = inputs[in_domains[0]].shape[0]
    
    masks = {
        "rgb": torch.zeros((B, num_patchs), dtype=torch.long) 
               if "rgb" in in_domains else torch.ones((B, num_patchs), dtype=torch.long),
        "depth": torch.zeros((B, num_patchs), dtype=torch.long) 
               if "depth" in in_domains else torch.ones((B, num_patchs), dtype=torch.long),
        "semseg": torch.zeros((B, num_patchs), dtype=torch.long) 
               if "semseg" in in_domains else torch.ones((B, num_patchs), dtype=torch.long)
    }
    
    masks = {k: torch.LongTensor(v).to(device) for k, v in masks.items()}
    
    preds, masks = model(
        inputs, 
        num_encoded_tokens=num_encoded_tokens, 
        semseg_gt=inputs["semseg"] if "semseg" in in_domains else None,
        in_domains=in_domains,
        semseg_stride=semseg_stride, 
        mask_inputs=True,
        task_masks=masks,
        return_embeddings=return_embed
    )

    return preds, masks