import open3d as o3d
import os


def load_pointcloud(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    print(pcd)
    return pcd


def visualize(pcd):
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    scene_path = r".\data\scannet\scene0000_00"
    ply_file = os.path.join(scene_path, "scene0000_00_vh_clean_2.ply")

    pcd = load_pointcloud(ply_file)
    visualize(pcd)