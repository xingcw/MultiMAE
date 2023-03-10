import os
import cv2
import glob
import shutil
import numpy as np
from PIL import Image
from pathlib import Path


def extract_data_from_racing(data_folder, target_folder, depth_format="tiff"):
    """Extracts data from the racing dataset and saves it in the desired format.

    Args:
        data_folder (str): source folder containing the data
        target_folder (str): target folder where the data should be saved
        depth_format (str, optional): desired depth image format. Defaults to "tiff".
    """
    total_num_imgs = 0

    for env_folder in sorted(os.listdir(data_folder)):
        for trial_folder in sorted(os.listdir(data_folder / env_folder)):
            sample_folder = data_folder / env_folder / trial_folder
            for file in sorted(glob.glob(str(sample_folder / "*.npz"))):
                
                split = "train" if total_num_imgs % 5 != 0 else "val"
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
                    normalized_depth = depth_img / depth_img.max()
                    depth_int16 = np.round(normalized_depth * 65535).astype(np.uint16)
                    cv2.imwrite(str(target_folder / f"{split}/depth/data/{total_num_imgs:06d}.png"), depth_int16)
                else:
                    raise KeyError(f"Depth format {depth_format} not supported.")
                    
                print(f"Copying {old_file_name} to {new_depth_path}")
                
                total_num_imgs += 1
                
    print(f"Total number of images: {total_num_imgs}")