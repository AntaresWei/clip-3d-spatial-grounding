import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
from models.clip_model import CLIPModel
from models.spatial_grounder import sliding_window_similarity
from PIL import Image

# 可视化相似度热力图

image = Image.open(r".\data\scannet\scene0000_00\color\000200.jpg")
text = "a black chair"

clip_model = CLIPModel()

heatmap = sliding_window_similarity(image, clip_model,text)

# heatmap = grounder.compute_heatmap(heatmap)

plt.imshow(heatmap, cmap='jet')
plt.colorbar()
plt.title("CLIP Similarity Heatmap")
plt.show()