import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.clip_model import CLIPModel
from PIL import Image

# 测试CLIP模型相似度计算

clip_model = CLIPModel()

image = Image.open(r".\data\scannet\scene0000_00\color\000200.jpg")

text = "a chair"

img_feat = clip_model.encode_image(image)
txt_feat = clip_model.encode_text(text)

score = clip_model.compute_similarity(img_feat, txt_feat)

print("Similarity:", score)