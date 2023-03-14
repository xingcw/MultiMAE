import os
import cv2
import glob
import shutil
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from pipelines.utils.constants import DEPTH_MAP_SCALE


def extract_data_from_racing(data_folder, target_folder, train_val_split=0.8, depth_format="png", sample_method="seq"):
    """Extracts data from the racing dataset and saves it in the desired format.

    Args:
        data_folder (str): source folder containing the data
        target_folder (str): target folder where the data should be saved
        train_val_split (float, optional): fraction of data to use for training. Defaults to 0.8.
        depth_format (str, optional): desired depth image format. Defaults to "png". options: ["png", "tiff"]
        sample_method (str, optional): method to sample the data. Defaults to "seq". options: ["seq", "random"]
    """
    total_num_imgs = 0
    
    if sample_method == "random":
        probs = np.random.uniform(0, 1, 2^16)
    else:
        val_freq = int(1 / (1 - train_val_split))
    
    assert os.path.exists(data_folder), f"Source folder {data_folder} does not exist."
    assert not os.path.exists(target_folder), f"Target folder {target_folder} already exists. Please delete it first."
    
    # prepare data folders
    os.makedirs(target_folder / "train/rgb/data")
    os.makedirs(target_folder / "train/depth/data")
    os.makedirs(target_folder / "train/semseg/data")
    os.makedirs(target_folder / "val/rgb/data")
    os.makedirs(target_folder / "val/depth/data")
    os.makedirs(target_folder / "val/semseg/data")

    for env_folder in sorted(os.listdir(data_folder)):
        for trial_folder in sorted(os.listdir(data_folder / env_folder)):
            sample_folder = data_folder / env_folder / trial_folder
            for file in sorted(glob.glob(str(sample_folder / "*.npz"))):
                
                if sample_method == "random":
                    split = "train" if probs[total_num_imgs] < train_val_split else "val"
                else:
                    split = "train" if total_num_imgs % val_freq != 0 else "val"
                    
                old_file_name = Path(file).stem.strip()
                # copy rgb image
                rgb_path = sample_folder / f"{old_file_name}_rgb.png"
                new_rgb_path = target_folder / f"{split}/rgb/data/{total_num_imgs:06d}.png"
                shutil.copy2(rgb_path, new_rgb_path)
                # print(f"Copying {old_file_name} to {new_rgb_path}")
                
                # copy semseg image
                semseg_path = sample_folder / f"{old_file_name}_semseg.png"
                new_semseg_path = target_folder / f"{split}/semseg/data/{total_num_imgs:06d}.png"
                shutil.copy2(semseg_path, new_semseg_path)
                # print(f"Copying {old_file_name} to {new_semseg_path}")
                
                # convert and copy depth image
                depth_path = sample_folder / f"{old_file_name}_depth.npy"
                depth = np.load(depth_path)
                
                if depth_format == "tiff":
                    depth_img = Image.fromarray(depth.squeeze())
                    new_depth_path = target_folder / f"{split}/depth/data/{total_num_imgs:06d}.tiff"
                    depth_img.save(new_depth_path)
                    
                elif depth_format == "png":
                    normalized_depth = depth / DEPTH_MAP_SCALE
                    normalized_depth = np.clip(normalized_depth, 0, 1)
                    depth_int16 = np.round(normalized_depth * 65535).astype(np.uint16)
                    new_depth_path = str(target_folder / f"{split}/depth/data/{total_num_imgs:06d}.png")
                    cv2.imwrite(new_depth_path, depth_int16)
                else:
                    raise KeyError(f"Depth format {depth_format} not supported.")
                    
                print(f"Copying {old_file_name} to {new_depth_path}")
                
                total_num_imgs += 1
                
    print(f"Total number of images: {total_num_imgs}")
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", type=str, default="splits", help="which track is the data from")
    parser.add_argument("--depth_format", type=str, default="png", help="desired depth image format")
    parser.add_argument("--sample_method", type=str, default="seq", help="method to sample the data")
    parser.add_argument("--train_val_split", type=float, default=0.8, help="fraction of data to use for training")
    parser.add_argument("--timestamp", type=str, default=None, help="timestamp of the data")
    args = parser.parse_args()
    
    data_lookup = {
        "splits": "SplitS_demo",
        "figure8": "Figure8_small_demo",
        "circle": "Circle_small_demo",
    }    
    
    flightmare_path = Path(os.environ["FLIGHTMARE_PATH"])
    multimae_path = flightmare_path.parent / "vision_backbones/MultiMAE"
    target_folder = multimae_path / "datasets/test"
    data_folder = flightmare_path / f"flightpy/datasets/{data_lookup[args.track]}/{args.timestamp}/data/data/epoch_0000"
    
    extract_data_from_racing(data_folder, target_folder, args.train_val_split, args.depth_format, args.sample_method)
