import torch
import random
from functools import partial
from pathlib import Path
from multimae.models.multimae import multivit_base
from multimae.models.input_adapters import PatchedInputAdapter, SemSegInputAdapter
from multimae.models.output_adapters import SpatialOutputAdapter
from multimae.tools.multimae2vit_converter import multimae_to_vit
from multimae.tools import multimae2vit_converter


device = "cuda:0"
fake_input = torch.rand(size=(1, 3, 224, 224))
inputs= {
    "rgb": fake_input.to(device)
}

DOMAIN_CONF = {
    'rgb': {
        'input_adapter': partial(PatchedInputAdapter, num_channels=3, stride_level=1),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=3, stride_level=1),
    },
    'depth': {
        'input_adapter': partial(PatchedInputAdapter, num_channels=1, stride_level=1),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=1, stride_level=1),
    },
    'semseg': {
        'input_adapter': partial(SemSegInputAdapter, num_classes=133,
                                 dim_class_emb=64, interpolate_class_emb=False, stride_level=4),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=133, stride_level=4),
    },
}
DOMAINS = ['rgb', 'depth', 'semseg']

input_adapters = {
    domain: dinfo['input_adapter'](
        patch_size_full=16,
    )
    for domain, dinfo in DOMAIN_CONF.items() if domain == "rgb"
}

pretrained_path = Path(__file__).parent / "pretrained_models/multimae-b_98_rgb+-depth-semseg_1600e_multivit-afff3f8c.pth"

state_dict = torch.load(pretrained_path, map_location=device)

model = multivit_base(input_adapters=input_adapters, output_adapters=None)
model.load_state_dict(state_dict["model"], strict=False)

model.to(device).eval()
with torch.no_grad():
    outputs = model(inputs)

print(outputs)
print(outputs.shape)