import os
import cv2
import glob
import shutil
import socket
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from multimae.utils.data_constants import GATE_SEMSEG_CLASS_ID
from pipelines.utils.constants import DEPTH_MAP_SCALE
from pipelines.utils.label_utils.semseg_from_corners import draw_polygon


def extract_data_from_racing(
        data_folder, 
        target_folder, 
        num_shift=0, 
        train_val_split=0.8, 
        depth_format="png", 
        sample_method="seq", 
        gate_corners=None
    ):
    """Extracts data from the racing dataset and saves it in the desired format.

    Args:
        data_folder (str): source folder containing the data
        target_folder (str): target folder where the data should be saved
        train_val_split (float, optional): fraction of data to use for training. Defaults to 0.8.
        depth_format (str, optional): desired depth image format. Defaults to "png". options: ["png", "tiff"]
        sample_method (str, optional): method to sample the data. Defaults to "seq". options: ["seq", "random"]
        gate_corners (np.array, optional): array containing the corners of the gate in each image. Defaults to None.
    """
    total_num_imgs = 0
    
    if sample_method == "random":
        probs = np.random.uniform(0, 1, 2**16)
    else:
        val_freq = int(1 / (1 - train_val_split))
    
    assert os.path.exists(data_folder), f"Source folder {data_folder} does not exist."
    # assert not os.path.exists(target_folder), f"Target folder {target_folder} already exists. Double check it."
    
    # prepare data folders
    os.makedirs(target_folder / "train/rgb/data", exist_ok=True)
    os.makedirs(target_folder / "train/depth/data", exist_ok=True)
    os.makedirs(target_folder / "train/semseg/data", exist_ok=True)
    os.makedirs(target_folder / "val/rgb/data", exist_ok=True)
    os.makedirs(target_folder / "val/depth/data", exist_ok=True)
    os.makedirs(target_folder / "val/semseg/data", exist_ok=True)
    
    use_fake_semseg = False
    
    if gate_corners is not None:
        os.makedirs(target_folder / "train/semseg_gt/data", exist_ok=True)
        os.makedirs(target_folder / "val/semseg_gt/data", exist_ok=True)
        use_fake_semseg = True
        
    for env_id, env_folder in tqdm(enumerate(sorted(os.listdir(data_folder)))):
        current_step_id = 0
        for trial_folder in sorted(os.listdir(data_folder / env_folder)):
            sample_folder = data_folder / env_folder / trial_folder
            for file in sorted(glob.glob(str(sample_folder / "*.npz"))):
                
                if sample_method == "random":
                    split = "train" if probs[total_num_imgs] < train_val_split else "val"
                else:
                    split = "train" if total_num_imgs % val_freq != 0 else "val"
                    
                old_file_name = Path(file).stem.strip()
                save_id = total_num_imgs + num_shift
                
                # copy rgb image
                rgb_path = sample_folder / f"{old_file_name}_rgb.png"
                new_rgb_path = target_folder / f"{split}/rgb/data/{save_id:06d}.png"
                shutil.copy2(rgb_path, new_rgb_path)
                # print(f"Copying {old_file_name} to {new_rgb_path}")
                
                # copy semseg image
                semseg_path = sample_folder / f"{old_file_name}_semseg.png"
                semseg_folder = "semseg_gt" if use_fake_semseg else "semseg"
                new_semseg_path = target_folder / f"{split}/{semseg_folder}/data/{save_id:06d}.png"
                shutil.copy2(semseg_path, new_semseg_path)
                # print(f"Copying {old_file_name} to {new_semseg_path}")
                
                # convert and copy depth image
                depth_path = sample_folder / f"{old_file_name}_depth.npy"
                depth = np.load(depth_path)
                
                if depth_format == "tiff":
                    depth_img = Image.fromarray(depth.squeeze())
                    new_depth_path = target_folder / f"{split}/depth/data/{save_id:06d}.tiff"
                    depth_img.save(new_depth_path)
                    
                elif depth_format == "png":
                    normalized_depth = depth / DEPTH_MAP_SCALE
                    normalized_depth = np.clip(normalized_depth, 0, 1)
                    depth_int16 = np.round(normalized_depth * 65535).astype(np.uint16)
                    new_depth_path = str(target_folder / f"{split}/depth/data/{save_id:06d}.png")
                    cv2.imwrite(new_depth_path, depth_int16)
                else:
                    raise KeyError(f"Depth format {depth_format} not supported.")
                    
                print(f"Copying {old_file_name} to {new_depth_path}")
                
                if gate_corners is not None:
                    # generate fake semseg image from corners
                    gate_corner = gate_corners[current_step_id, env_id, :, :, :]
                    img = Image.new('L', (depth.shape[1], depth.shape[0]), 0)
                    
                    for i in range(gate_corner.shape[0]):
                        corners = gate_corner[i].flatten().reshape(4, 2, order='F')
                        img = draw_polygon(corners, 4, img=img, fill_value=GATE_SEMSEG_CLASS_ID)
                        
                    img_save_path = target_folder / f"{split}/semseg/data/{save_id:06d}.png"
                    img.save(img_save_path, 'PNG')
                    
                    print(f"Generate fake semseg image and saved to {img_save_path}")
                
                total_num_imgs += 1
                current_step_id += 1
                
    print(f"Total number of images: {total_num_imgs}")
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", "-t", type=str, default="splits", help="which track is the data from")
    parser.add_argument("--depth_format", "-d", type=str, default="png", help="desired depth image format")
    parser.add_argument("--sample_method", "-s", type=str, default="random", help="method to sample the data")
    parser.add_argument("--train_val_split", "-tv", type=float, default=0.8, help="fraction of data to use for training")
    parser.add_argument("--num_shift", "-ns", type=int, default=0, help="number of images to shift the ids")
    parser.add_argument("--timestamp", "-ts", type=str, default=None, help="timestamp of the data")
    args = parser.parse_args()
    
    data_lookup = {
        "splits": "SplitS_demo",
        "figure8": "Figure8_small_demo",
        "circle": "Circle_small_demo",
    }    
    
    flightmare_path = Path(os.environ["FLIGHTMARE_PATH"])
    multimae_path = flightmare_path.parent / "vision_backbones/MultiMAE"
    dataset_folder = flightmare_path / f"flightpy/datasets"
    
    server = socket.gethostname()
    if server == "snaga":
        dataset_folder = Path("/data/storage/chunwei/multimae/datasets/raw_data_tracks_all")
        multimae_path = Path("/data/storage/chunwei/multimae")
    
    data_folder = dataset_folder / f"{data_lookup[args.track]}/{args.timestamp}"
    quad_state_path = data_folder / "quad_states.npy"
    gate_corners_path = data_folder / "gate_corners.npy"

    if os.path.exists(gate_corners_path):
        gate_corners = np.load(gate_corners_path)
    else:
        gate_corners = None
        
    target_folder = multimae_path / "datasets/test_fake_semseg"
    data_folder = data_folder / "data/data/epoch_0000"
    
    extract_data_from_racing(
        data_folder=data_folder,
        target_folder=target_folder,
        num_shift=args.num_shift,
        train_val_split=args.train_val_split,
        depth_format=args.depth_format,
        sample_method=args.sample_method,
        gate_corners=gate_corners
    )
