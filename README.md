# Open-Vocabulary 3D Semantic Grounding (CLIP + ScanNet)

A multi-view 3D semantic grounding system that aligns natural language queries with indoor RGB-D scenes and localizes target objects in 3D point clouds.

This project integrates vision-language models with geometric projection and multi-view statistical fusion to enable text-driven 3D object localization.

---

## 🚀 Overview

Given a natural language query (e.g., *"a guitar"*, *"a white toilet in the bathroom"*),  
the system:

1. Computes semantic similarity using CLIP
2. Generates 2D semantic heatmaps via sliding-window matching
3. Projects 2D semantic responses into 3D space using RGB-D geometry
4. Aggregates multi-view semantic scores across frames
5. Applies statistical filtering (percentile-based thresholding)
6. Outputs a highlighted 3D semantic point cloud

---

## 🧠 Key Features

- **Open-vocabulary 3D localization** via CLIP
- **Multi-view semantic fusion** to improve robustness
- **Adaptive percentile thresholding (p98 / p99)** for noise suppression
- **Frame gating mechanism** to prevent semantic dilution
- **Efficient KD-Tree nearest-neighbor fusion**
- Compatible with ScanNet RGB-D scenes

---

## 🏗 System Pipeline

``` TypeScript
Text Query
↓
CLIP Text Encoder
↓
Sliding Window Image-Text Similarity
↓
2D Semantic Heatmap
↓
RGB-D Backprojection (Camera Intrinsics + Pose)
↓
Multi-view Score Aggregation
↓
Percentile-based Filtering
↓
3D Semantic Point Cloud
```


---

## 🔍 Technical Challenges & Solutions

### 1. CLIP is image-level, not pixel-level

- Implemented sliding-window semantic matching
- Generated dense heatmaps from patch-level responses

### 2. Multi-view semantic dilution

Simple averaging across frames suppresses true signals.

**Solution:**
- Frame-level gating
- Observation-count constraint
- Percentile-based adaptive thresholding

### 3. Noise propagation across frames

**Solution:**
- Valid observation filtering (`score_count >= k`)
- High-percentile selection (p98 / p99)
- Optional top-K fallback

---

## 📊 Statistical Modeling Strategy

Instead of fixed absolute thresholds, this project uses:

```python
p99 = np.percentile(final_score[valid], 99)
semantic_mask = final_score > p99
```

This approach:

Adapts to scene-dependent score distributions

Preserves right-tail semantic signals

Reduces manual tuning

## 🧪 Example Results

Strong localization for distinctive objects (e.g., guitar)

Stable right-tail score distribution across views

Robust suppression of background noise

## 📂 Dataset

ScanNet RGB-D dataset

Uses intrinsic/extrinsic camera parameters

Supports multi-frame scene processing

## 🛠 Dependencies

``` TypeScript
Python 3.10

PyTorch

Open3D

NumPy

CLIP (OpenAI)
```

## 📈 Future Improvements

Attention-based patch extraction (ViT attention maps)

Positive-negative contrastive scoring

SAM-based region refinement

3D bounding box extraction via clustering

Occlusion-aware visibility modeling

## 🎯 Applications

Open-vocabulary 3D scene understanding

Vision-language robotics

Text-driven 3D search

Multi-modal spatial grounding research
