import torch
import clip
from PIL import Image
import numpy as np


class CLIPModel:

    def __init__(self, device="cuda"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device, download_root=r".\models\clip")

    def encode_text(self, text):
        text_tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def encode_image(self, image):
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def compute_similarity(self, image_features, text_features):
        similarity = (image_features @ text_features.T).squeeze()
        return similarity.item()