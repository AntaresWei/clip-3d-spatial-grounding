import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import open3d as o3d
from PIL import Image

from datasets.scannet_dataset import ScanNetScene
from models.clip_model import CLIPModel
from models.spatial_grounder import sliding_window_similarity, heatmap_to_mask
from geometry.projection import mask_to_world_points


scene_path = r".\data\scannet\scene0000_00"
scene = ScanNetScene(scene_path)

frame_id = scene.frame_ids[0]
color, depth, pose = scene.load_frame(frame_id)
intrinsic = scene.load_intrinsic()

depth_m = depth.astype(np.float32) / 1000.0

# ---- CLIP ----
clip_model = CLIPModel()

image = Image.fromarray(color[:, :, ::-1])
# text = "a black chair"
text = "Grey sofa"

text_feat = clip_model.encode_text(text)
heatmap = sliding_window_similarity(image, clip_model, text_feat)

mask = heatmap_to_mask(heatmap, percentile=92)

# ---- 投影 ----
points = mask_to_world_points(mask, depth_m, intrinsic, pose)

# ---- 可视化 ----
pcd_semantic = o3d.geometry.PointCloud()
pcd_semantic.points = o3d.utility.Vector3dVector(points)
pcd_semantic.paint_uniform_color([1, 0, 0])  # 红色

# 加载原始点云
ply_path = scene_path + r"\scene0000_00_vh_clean_2.ply"
pcd_scene = o3d.io.read_point_cloud(ply_path)
pcd_scene.paint_uniform_color([0.7, 0.7, 0.7])

o3d.visualization.draw_geometries([pcd_scene, pcd_semantic])