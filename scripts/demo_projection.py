import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import open3d as o3d
from datasets.scannet_dataset import ScanNetScene
from geometry.projection import depth_to_world_points

scene = ScanNetScene(r".\data\scannet\scene0000_00")

frame_id = scene.frame_ids[0]
color, depth, pose = scene.load_frame(frame_id)
intrinsic = scene.load_intrinsic()

depth_m = depth.astype(np.float32) / 1000.0

points = depth_to_world_points(depth_m, intrinsic, pose)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

o3d.visualization.draw_geometries([pcd])