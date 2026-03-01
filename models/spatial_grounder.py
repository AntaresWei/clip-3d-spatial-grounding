import numpy as np
from PIL import Image


def sliding_window_similarity(image, clip_model, text_feat, window_size=224, stride=112):
    '''
    返回相似度热力图
    '''
    image_np = np.array(image)
    H, W, _ = image_np.shape

    heatmap = np.zeros((H, W))
    count_map = np.zeros((H, W))  # 新增

    for y in range(0, H - window_size + 1, stride):
        for x in range(0, W - window_size + 1, stride):

            crop = image_np[y:y+window_size, x:x+window_size]
            crop_img = Image.fromarray(crop)

            img_feat = clip_model.encode_image(crop_img)
            sim = (img_feat @ text_feat.T).item()

            heatmap[y:y+window_size, x:x+window_size] += sim
            count_map[y:y+window_size, x:x+window_size] += 1

    heatmap = heatmap / (count_map + 1e-6)  # 关键

    return heatmap

def heatmap_to_mask(heatmap, threshold=0.30, max_ratio=0.05):
    '''
    返回热力图的二值掩码
    '''
    # threshold = np.percentile(heatmap, percentile)
    mask = heatmap > threshold
    # 防止面积过大
    max_pixels = int(max_ratio * heatmap.size)

    if mask.sum() > max_pixels:
        flat = heatmap.flatten()
        topk_idx = np.argsort(flat)[-max_pixels:]
        new_mask = np.zeros_like(flat, dtype=bool)
        new_mask[topk_idx] = True
        mask = new_mask.reshape(heatmap.shape)
    return mask
