# CoVT Reasoner

The official training code for **CoVT-Reasoner** with Necessity Reward mechanism, based on **"[VisionReasoner: Unified Visual Perception and Reasoning via Reinforcement Learning](https://arxiv.org/pdf/2505.12081)"**.


**Key Features**:

1. Based on [EasyR1](https://github.com/hiyouga/EasyR1) and [veRL](https://github.com/volcengine/verl), supporting model split during sampling for better GPU memory efficiency
2. Supporting Qwen2.5-VL series models with visual feature extraction capabilities
3. Multi-dimensional reward design: format, accuracy, reasoning quality, and feature usage

## Contents
- [Installation](#installation)
- [Training](#training)



## Installation

```bash
conda create -n covt_reasoner python=3.12
conda activate covt_reasoner
pip install torch==2.6.0 torchvision==0.21.0
pip install -e .
```

## Training

### CoVT-Reasoner RL Training with Necessity Reward

> [!NOTE]
> The recommanded training requirement for 7B model is **8 GPUs** (tested on 8x46G or 8x80G).


#### Training Data Preparation

Training Data: **ViRL39K** dataset located at `https://huggingface.co/datasets/TIGER-Lab/ViRL39K/`

> [!TIP]
> The training data should contain visual reasoning tasks with ground truth answers. Each sample will be used to generate 8 rollouts for group-wise reward computation.




#### Start Training

Run the training script:
```bash
bash training_scripts/run_visionreasoner_7b_rl39k_allpenalty.sh
```

Merge Checkpoint in Hugging Face Format

```bash
python3 training_scripts/model_merger.py --local_dir [path_to_your_actor_checkpoint]
```