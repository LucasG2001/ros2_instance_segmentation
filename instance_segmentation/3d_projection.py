import warnings
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
from lang_sam import LangSAM
import time
import random
import cv2
import open3d as o3d
from test import *
import os


def create_pointcloud_from_mask(color_image, depth_image, intrinsic, mask):
    """
    Create a colored point cloud from color and depth images using Open3D.

    Args:
    - color_image (np.array): The color image as a numpy array.
    - depth_image (np.array): The depth image as a numpy array.
    - intrinsic (o3d.camera.PinholeCameraIntrinsic): Camera intrinsic parameters.

    Returns:
    - o3d.geometry.PointCloud: The resulting colored point cloud.
    """
    masked_color_image = color_image
    masked_depth_image = mask * depth_image

    # Convert images to Open3D format
    color_image_o3d = o3d.geometry.Image(cv2.cvtColor(masked_color_image, cv2.COLOR_BGR2RGB))
    depth_image_o3d = o3d.geometry.Image(masked_depth_image.astype(np.float32))

    # Create RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image_o3d, depth_image_o3d, convert_rgb_to_intensity=False)

    # Create point cloud from RGBD image
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    # Subsampling
    pcd = pcd.uniform_down_sample(every_k_points=3)  # Use every second point
    o3d.visualization.draw_geometries([pcd], window_name="after uniform sample")
    # Stochastic outlier removal
    pcd_downsampled, _ = pcd_downsampled.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
    o3d.visualization.draw_geometries([pcd_downsampled], window_name="after statistical outlier filtering")
    return pcd_downsampled



if __name__ == "__main__":
    # load color and depth image
    color_img_path = "images/color_image1.png"
    depth_img_path = "images/depth_image1.png"
    if os.path.exists(color_img_path):
        print("path exists")
    else:
        print("path does not exist")
    color_image1 = cv2.imread(color_img_path)
    depth_image1 = cv2.imread(depth_img_path, -1) / 10000
    image_pil = Image.open(color_img_path).convert("RGB")
    text_prompt = "the robot on the table"
    # load camera intrinsic
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=1280, height=720,
                                                        fx=533.77, fy=533.53,
                                                        cx=661.87, cy=351.29)

    #load model
    model = LangSAM(sam_type="vit_b")   
    masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
    if len(masks) == 0:
        print(f"No objects of the '{text_prompt}' prompt detected in the image.")
    else:
        # Convert masks to numpy arrays
        masks_np = [mask.squeeze().cpu().numpy() for mask in masks]
        mask = masks_np[0]

    # Create point cloud
        point_cloud = create_pointcloud_from_mask(color_image1, depth_image1, o3d_intrinsic, mask)

        # Visualize the point cloud
        o3d.visualization.draw_geometries([point_cloud])
