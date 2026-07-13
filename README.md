# LDCNet: An Energy-guided Attention and Multi-scale Cross-stage Partial Network for Lightweight Steel Surface Defect Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official PyTorch implementation of the paper:
**LDCNet: An Energy-guided Attention and Multi-scale Cross-stage Partial Network for Lightweight Steel Surface Defect Detection** Zhentao Hua, Jing Yan*, Jiaxin Zou, Jingchang Hu, Bingqing Wang  
*School of Artificial Intelligence, Henan University, Zhengzhou, Henan, China*

---

## 🛠️ Environment and Dependencies

This project was developed and evaluated under the following environment settings:
* **OS:** Windows
* **CPU:** Intel Core i7-11700K
* **GPU:** NVIDIA GeForce RTX 3090 (24 GB VRAM)

### Software Dependencies
* **Python:** 3.9
* **PyTorch:** 1.12.1
* **CUDA:** 11.6

---

## 📌 Highlights

- 🎯 **Specialized for Edge-Side Steel Inspection** Specifically engineered to overcome low-contrast defect signatures, severe background noise, and extreme scale variations in industrial high-speed rolling lines.

- 🧩 **Modular and Lightweight Architecture** Breaks the conventional accuracy-efficiency trade-off using three synergistic components:
  * **EGA-PTB (Energy-Guided Ghost Attention Partial Transformer Block):** Combines lightweight Ghost convolutions with a parameter-free SimAM energy attention mechanism to amplify low-contrast visual clues.
  * **CSP-MSCB (Cross-Stage Partial Multi-Scale Convolutional Block):** Employs a parallel multi-branch depthwise-separable topology with channel shuffle to capture extreme morphological deformations.
  * **CS-LQE (Computation-Shared Lightweight Quality Estimation Head):** Establishes an explicit cross-task interaction loop to resolve the misalignment between classification and localization while cutting convolutional overhead by nearly 70%.

- ⚡ **State-of-the-Art Performance & High Throughput** Achieves highly competitive accuracy on public benchmarks with an ultra-compact deployment footprint:
  * **NEU-DET:** **79.4% mAP** / **81.1% Precision** at **173 FPS**.
  * **GC10-DET:** **64.0% mAP** / **65.3% Recall** at **207 FPS**.
  * **Model Footprint:** Requires only **1.6M parameters** and **4.7G FLOPs**, making it instantly deployable on resource-constrained edge hardware.

---

## 📁 Project Structure

```plaintext
LDCNet/
├── docker/               # Containerization configurations
├── docs/                 # Documentation and analysis files
├── examples/             # Inference and visualization examples
├── tests/                # Unit tests for architectural modules
├── ultralytics/          # Core model definitions and baseline adaptations
├── train.py              # Main training script
├── dataset.yaml          # Data configuration paths (NEU-DET / GC10-DET)
├── ldcnet.yaml           # Model network configuration profile
├── README.md             # Project overview
└── LICENSE               # MIT License
