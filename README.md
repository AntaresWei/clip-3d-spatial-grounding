# clip-3d-spatial-grounding
基于多模态大模型的「3D 场景语义定位与问答」：构建基于 CLIP / LLaVA 的多模态空间理解系统，实现自然语言驱动的 3D 场景目标定位与空间关系推理。

# CLIP-based 3D Spatial Grounding

This project explores text-driven 3D scene understanding using CLIP and ScanNet.
Given a natural language query, the system localizes relevant regions in a 3D point
cloud by aligning 2D visual features with text embeddings and projecting them into
3D space.

## Features
- CLIP-based vision-language alignment
- 2D-to-3D semantic projection
- Open3D visualization for spatial grounding

## Dataset
- ScanNet v2

## Pipeline
Text → CLIP → 2D region matching → 3D projection → visualization
