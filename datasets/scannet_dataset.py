import os
import cv2
import numpy as np


class ScanNetScene:

    def __init__(self, scene_path):
        self.scene_path = scene_path
        self.color_path = os.path.join(scene_path, "color")
        self.depth_path = os.path.join(scene_path, "depth")
        self.pose_path = os.path.join(scene_path, "pose")
        self.intrinsic_path = os.path.join(scene_path, "intrinsics_depth.txt")

        self.frame_ids = sorted([
            f.split('.')[0]
            for f in os.listdir(self.color_path)
            if f.endswith(".jpg")
        ])

    def load_frame(self, frame_id):
        color = cv2.imread(os.path.join(self.color_path, f"{frame_id}.jpg"))
        depth = cv2.imread(os.path.join(self.depth_path, f"{frame_id}.png"), -1)
        pose = np.loadtxt(os.path.join(self.pose_path, f"{frame_id}.txt"))

        return color, depth, pose

    def load_intrinsic(self):
        return np.loadtxt(self.intrinsic_path)