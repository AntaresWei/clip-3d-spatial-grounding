import numpy as np
from PIL import Image


def sliding_window_similarity(image, clip_model, text, window_size=224, stride=112):
    '''
    返回相似度热力图
    '''
    image_np = np.array(image)
    H, W, _ = image_np.shape

    heatmap = np.zeros((H, W))
    
    text_feat = clip_model.encode_text(text)

    for y in range(0, H - window_size, stride):
        for x in range(0, W - window_size, stride):

            crop = image_np[y:y+window_size, x:x+window_size]
            crop_img = Image.fromarray(crop)

            img_feat = clip_model.encode_image(crop_img)
            
            sim = (img_feat @ text_feat.T).item()

            heatmap[y:y+window_size, x:x+window_size] += sim

    return heatmap

def generate_mask(heatmap, percentile=90):
    '''
    返回热力图的二值掩码
    '''
    threshold = np.percentile(heatmap, percentile)
    mask = heatmap > threshold
    return mask
