import warnings
import numpy as np
import matplotlib.pyplot as plt
from instance_segmentation.test import display_image_with_masks
from PIL import Image
from io import BytesIO
from lang_sam import LangSAM
import time
import random
import cv2
import open3d as o3d
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
import os

class PointCloudNode(Node):
    def __init__(self):
        super().__init__('pointcloud_node')
        self.subscription = self.create_subscription(
            ROSImage,
            'color_image',
            self.color_image_callback,
            10)
        self.subscription = self.create_subscription(
            ROSImage,
            'depth_image',
            self.depth_image_callback,
            10)
        self.bridge = CvBridge()
        self.color_image = None
        self.depth_image = None
        self.model = LangSAM(sam_type="vit_b")
        self.text_prompt = None
        self.o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=1280, height=720,
                                                               fx=533.77, fy=533.53,
                                                               cx=661.87, cy=351.29)

    def color_image_callback(self, msg):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def depth_image_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1") / 10000  # Assuming depth is in meters
        self.process_images()

    def process_images(self, visualize_masks="True"):
        if self.color_image is not None and self.depth_image is not None:
            image_pil = Image.fromarray(cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB))
            masks, boxes, phrases, logits = self.model.predict(image_pil, self.text_prompt)
            if len(masks) == 0:
                self.get_logger().info(f"No objects of the '{self.text_prompt}' prompt detected in the image.")
                return
            masks_np = [mask.squeeze().cpu().numpy() for mask in masks]
            mask = masks_np[0]
            if visualize_masks:
                display_image_with_masks(self.color_image, masks_np)
            point_cloud = self.create_pointcloud_from_mask(self.color_image, self.depth_image, self.o3d_intrinsic, mask)

    def create_pointcloud_from_mask(self, color_image, depth_image, intrinsic, mask):
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
        pcd = pcd.uniform_down_sample(every_k_points=2)  # Use every second point
        o3d.visualization.draw_geometries([pcd], window_name="after uniform sample")
        # Voxel downsampling
        voxel_size = 0.000001  # 1 cm voxel size
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
        o3d.visualization.draw_geometries([pcd_downsampled], window_name="after voxel sample")

        # Stochastic outlier removal
        pcd_downsampled, _ = pcd_downsampled.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
        o3d.visualization.draw_geometries([pcd_downsampled], window_name="after statistical outlier filtering")
        return pcd_downsampled

def main(args=None):
    rclpy.init(args=args)
    pointcloud_node = PointCloudNode()
    color_path = "/home/lucas/franka_ros2_ws/src/instance_segmentation/instance_segmentation/images/color_image1.png"
    pointcloud_node.color_image = cv2.imread(color_path)
    if os.path.exists(color_path):
        print("path exists")
    else:
        print("path does not exist")
    pointcloud_node.depth_image = cv2.imread("/home/lucas/franka_ros2_ws/src/instance_segmentation/instance_segmentation/images/depth_image1.png", -1) / 10000

    while rclpy.ok():
        # show color IMage
        window_name = "color image input"
        print(window_name)
        cv2.imshow(window_name, pointcloud_node.color_image)
        # waits for user to press any key (this is necessary to avoid Python kernel form crashing) 
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        # Gather prompt input
        pointcloud_node.text_prompt = input("Enter text prompt (or press 'x' to exit): ")
        if pointcloud_node.text_prompt.lower() == 'x':
            break
        
        if key == ord('x'):
            break

        pointcloud_node.process_images(visualize_masks=True)
        rclpy.spin_once(pointcloud_node, timeout_sec=0.1)  # Process one cycle of callbacks

    pointcloud_node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
