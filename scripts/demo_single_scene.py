import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.scannet_dataset import ScanNetScene
import matplotlib.pyplot as plt

scene = ScanNetScene(r".\data\scannet\scene0000_00")

print("Total frames:", len(scene.frame_ids))

color, depth, pose = scene.load_frame(scene.frame_ids[0])

print("Color shape:", color.shape)
print("Depth shape:", depth.shape)
print("Pose shape:", pose.shape)

plt.imshow(color[:, :, ::-1])
plt.title("RGB Frame")
plt.show()
