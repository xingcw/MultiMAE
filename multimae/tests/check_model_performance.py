# To supress DPT and Mask2Former warnings
import warnings
warnings.filterwarnings("ignore")

import os
import cv2

import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torchvision.transforms.functional as TF

# DPT dependencies for depth pseudo labeling
from pipelines.utils.data_utils import preprocess_multimae_inputs_in_batch
from multimae.utils.plot_utils import plot_predictions, get_semseg_metadata
from multimae.models.multimae import pretrain_multimae_base
from multimae.tools.load_multimae import load_model, multimae_predict


def test_model(model_name, data_folder, pred_save_folder, file_name, metadata, device="cuda"):
    
    sample_rgb = data_folder / f"rgb/data/{file_name.stem}.png"
    sample_depth = data_folder / f"depth/data/{file_name.stem}.png"
    sample_semseg = data_folder / f"semseg/data/{file_name.stem}.png"

    image = cv2.imread(str(sample_rgb))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_float = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
    c, h, w = image_float.shape

    image_float = TF.center_crop(image_float, min([h, w]))
    image_float = TF.resize(image_float, 224)
    image_float = image_float.permute(1, 2, 0).numpy().astype(np.float32)

    image = Image.fromarray(image)

    semseg = cv2.imread(str(sample_semseg))
    depth = cv2.imread(str(sample_depth), cv2.IMREAD_ANYDEPTH)
    depth = depth / 65535.0
    
    input_dict = {
        "rgb": np.expand_dims(image, [0, -1]),
        "depth": np.expand_dims(depth, [0, 3, 4]),
        "semseg": np.expand_dims(semseg[:, :, 0], [0, 3, 4])
    }

    inputs = preprocess_multimae_inputs_in_batch(input_dict, batch_axis=True, hist_axis=True)
    inputs = {k: v.squeeze(-1).to(device) for k,v in inputs.items()}
    in_domains = ["semseg"]

    patch_x, patch_y = 14, 14
    num_patchs = patch_x * patch_y
    num_encoded_tokens = num_patchs * len(in_domains)
    B = inputs[in_domains[0]].shape[0]

    masks = {
        "rgb": torch.ones((B, num_patchs), dtype=torch.long),
        "depth": torch.ones((B, num_patchs), dtype=torch.long),
        "semseg": torch.zeros((B, num_patchs), dtype=torch.long),
    }

    masks = {k: torch.LongTensor(v).to(device) for k, v in masks.items()}

    preds, masks = model(
        inputs, 
        num_encoded_tokens=num_encoded_tokens, 
        semseg_gt=inputs["semseg"],
        in_domains=in_domains,
        semseg_stride=4, 
        mask_inputs=True,
        task_masks=masks,
        return_embeddings=False
    )

    preds = {domain: pred.detach().cpu() for domain, pred in preds.items()}
    masks = {domain: mask.detach().cpu() for domain, mask in masks.items()}

    res = plot_predictions(inputs, preds, masks, metadata=metadata, return_fig=True)
    res_path = pred_save_folder / model_name / f"{sample_rgb.stem}.png"
    res.save(res_path)
    
    print(f"Saved to {res_path}!")


if __name__ == "__main__":
    
    torch.set_grad_enabled(False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(1) # change seed to resample new mask

    model_name = "rgb-semseg_semseg-clean"
    model, args = load_model(model_name)
    print(model.output_adapters.keys())
    
    flightmare_path = Path(os.environ["FLIGHTMARE_PATH"])
    data_folder = flightmare_path.parent / "vision_backbones/MultiMAE/datasets/new_env/val"
    pred_save_folder = flightmare_path.parent / "vision_backbones/MultiMAE/results/predictions"
    
    os.makedirs(pred_save_folder / model_name, exist_ok=True)
    
    metadata = get_semseg_metadata(data_folder)
    
    for img_file in sorted(data_folder.glob("rgb/data/*.png")):
        test_model(model_name, data_folder, pred_save_folder, img_file, metadata, device)
