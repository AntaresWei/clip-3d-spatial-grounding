Open-Vocabulary 3D Semantic Grounding (CLIP + ScanNet)

A multi-view 3D semantic grounding system that aligns natural language queries with RGB-D indoor scenes.
The project bridges Vision-Language models and 3D geometry by projecting 2D semantic responses into 3D space and performing multi-view fusion.

🚀 Overview

Given a natural language query (e.g., "a guitar"), the system:

Computes 2D semantic heatmaps using CLIP.

Projects semantic responses into 3D point cloud space via RGB-D geometry.

Aggregates multi-view confidence scores across frames.

Applies statistical filtering (percentile-based thresholding).

Outputs semantically grounded 3D point clouds.

This enables open-vocabulary object localization in 3D scenes without task-specific training.

🧠 Key Contributions

Open-vocabulary 3D grounding using CLIP embeddings

RGB-D pixel-level back-projection using camera intrinsics & extrinsics

Multi-view semantic score aggregation

Frame gating mechanism to suppress noise propagation

Percentile-based adaptive thresholding (p98 / p99)

KD-Tree based 2D–3D nearest-neighbor semantic fusion

Statistical analysis of semantic score distributions

🏗️ System Pipeline

Text → CLIP → 2D Heatmap → Depth Back-Projection →
3D World Coordinates → Multi-View Score Fusion →
Percentile Filtering → 3D Semantic Visualization

🔍 Core Challenges
1. CLIP is image-level, not pixel-level

Sliding-window inference is used to approximate spatial semantic localization.

2. Multi-view score dilution

Naïve averaging causes semantic signal suppression due to non-visible frames.

3. Threshold instability

Absolute thresholds are unreliable across scenes.
Percentile-based filtering stabilizes right-tail semantic selection.

📊 Multi-View Fusion Strategy

For each 3D point:

final_score = score_sum / score_count

Enhancements include:

Frame-level gating (skip low-response frames)

Minimum observation constraint (score_count ≥ 2)

Percentile-based filtering (p98 / p99)

Optional top-k fallback selection

This reduces noise amplification and improves semantic stability.

📦 Project Structure
.
├── scripts/
│   ├── demo_grounding.py
│   ├── spatial_grounder.py
│   ├── scannet_dataset.py
├── third_party/
│   └── CLIP/
├── outputs/
└── README.md
🛠️ Installation
Requirements

Python 3.10

PyTorch (CUDA 12.1 recommended)

Open3D

NumPy

Scikit-learn

Setup
conda create -n clip3d python=3.10
conda activate clip3d

pip install torch torchvision
pip install open3d numpy scikit-learn

Download ScanNet dataset and configure path in scannet_dataset.py.

▶️ Usage
python scripts/demo_grounding.py

Modify the text query inside:

text = "a guitar"
📈 Example Results

The system successfully localizes:

Guitar (high separability class)

Toilet (higher semantic confusion with sink/bathtub)

Observations:

Object categories with strong visual distinctiveness produce clearer right-tail score distributions.

White indoor objects exhibit embedding overlap in CLIP space.

🧪 Experimental Notes

Score distribution example:

p90  ≈ 0.24
p95  ≈ 0.25
p97  ≈ 0.26
p99  ≈ 0.27
max  ≈ 0.29

Right-tail separation determines localization stability.

🔬 Future Improvements

Positive-negative contrast scoring

Visibility-aware updates

SAM-assisted region masking

DINOv2 backbone replacement

3D clustering for bounding box generation

Quantitative IoU evaluation
