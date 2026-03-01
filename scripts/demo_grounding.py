import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import open3d as o3d
from PIL import Image

from datasets.scannet_dataset import ScanNetScene
from models.clip_model import CLIPModel
from models.spatial_grounder import sliding_window_similarity
from geometry.projection import backproject_pixel_to_camera, camera_to_world


scene_path = r".\data\scannet\scene0000_00"
scene = ScanNetScene(scene_path)

text = "a bicycle"
frame_score_threshold = 0.26
# final_score_threshold = 0.275  # 必须 < final_score_max

clip_model = CLIPModel()
text_feat = clip_model.encode_text(text)

ply_path = scene_path + r"\scene0000_00_vh_clean_2.ply"
pcd_scene = o3d.io.read_point_cloud(ply_path)
scene_points = np.asarray(pcd_scene.points)
N = len(scene_points)
score_sum = np.zeros(N)
score_count = np.zeros(N)

kdtree = o3d.geometry.KDTreeFlann(pcd_scene)

intrinsic = scene.load_intrinsic()

for frame_id in scene.frame_ids[::1]:
    color, depth, pose = scene.load_frame(frame_id)
    depth_m = depth.astype(np.float32) / 1000.0
    image = Image.fromarray(color[:, :, ::-1])
    heatmap = sliding_window_similarity(image, clip_model, text_feat)
    if heatmap.max() < frame_score_threshold:
        continue

    H, W = depth_m.shape
    for v in range(0, H, 4):
        for u in range(0, W, 4):
            z = depth_m[v, u]
            if z <= 0:
                continue
            sim = heatmap[v, u]
            point_cam = backproject_pixel_to_camera(u, v, z, intrinsic)
            point_world = camera_to_world(point_cam, pose)[:3]
            _, idx, _ = kdtree.search_knn_vector_3d(point_world, 1)
            idx = idx[0]
            score_sum[idx] += sim
            score_count[idx] += 1

final_score = np.zeros(N)
valid = score_count >= 3
final_score[valid] = score_sum[valid] / score_count[valid]

# 用全局统计来选 absolute threshold
vals = final_score[valid]
print("valid_points", int(valid.sum()))
print("final_score_max", float(final_score.max()))
print("final_score_mean", float(final_score.mean()))

print("p90,p95,p97,p99:",
      np.percentile(vals, 90),
      np.percentile(vals, 95),
      np.percentile(vals, 97),
      np.percentile(vals, 99))


p99 = np.percentile(vals, 99)
print("p99:", float(p99))
semantic_mask = final_score > p99
print("selected_points", int(np.count_nonzero(semantic_mask)))

# semantic_mask = final_score > final_score_threshold


# 兜底用。如果阈值选不到点就取top k
if semantic_mask.sum() == 0:
    k = 500  # 调试用
    idx_top = np.argsort(final_score)[-k:]
    semantic_mask = np.zeros(N, dtype=bool)
    semantic_mask[idx_top] = True
    print("fallback_points", int(np.count_nonzero(semantic_mask)))


colors = np.tile(np.array([[0.7, 0.7, 0.7]]), (N, 1))
colors[semantic_mask] = [1, 0, 0]
pcd_scene.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([pcd_scene])