# CoVT Reasoner
The codebase for **CoVT-Reasoner**, including both **Supervised Fine-Tuning (SFT)** (ILVR framework) and **Reinforcement Learning (RL)** with Necessity Reward mechanism. This work is built upon **"[VisionReasoner: Unified Visual Perception and Reasoning via Reinforcement Learning](https://arxiv.org/pdf/2505.12081)"** and **"[Interleaved Latent Visual Reasoning with Selective Perceptual Modeling](https://arxiv.org/abs/2512.05665)"**.

## Key Features
1. **Dual Training Paradigm**: Supports both SFT (ILVR) and RL training with Necessity Reward
2. **Efficient Inference**: Based on [EasyR1](https://github.com/hiyouga/EasyR1) and [veRL](https://github.com/volcengine/verl), supporting model split during sampling for better GPU memory efficiency
3. **Multi-Modal Support**: Optimized for Qwen2.5-VL series models with visual feature extraction capabilities
4. **Multi-dimensional Reward (RL)**: Format, accuracy, reasoning quality, and feature usage
5. **Interleaved Reasoning (SFT)**: Based on CoMT dataset with selective perceptual modeling

## Contents
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
  - [SFT Training (ILVR)](#sft-training-ilvr)
  - [RL Training with Necessity Reward](#rl-training-with-necessity-reward)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Installation
### 1. Environment Setup
The code is tested with Python 3.11/3.12 (compatible with both versions). We recommend using Conda for environment management.

```bash
# Create conda environment (choose a name, e.g., covt_reasoner)
conda create -n covt_reasoner python=3.11
conda activate covt_reasoner

# Install PyTorch (compatible with both training paradigms)
pip install torch==2.6.0 torchvision==0.21.0

# Install SFT dependencies
pip install -r requirements.txt

# Install custom Transformers library (required for ILVR/SFT)
cd transformers
pip install -e .
cd ..

# Install RL dependencies (CoVT-Reasoner core)
pip install -e .
```

### 2. Accelerate Configuration
This project uses HuggingFace `accelerate` for distributed training (required for both SFT and RL).

```bash
accelerate config
```

## Data Preparation
### For SFT Training (ILVR)
We use the CoMT (Chain of Multi-modal Thought) dataset as the default training data for SFT:

1. Download processed data from [shuai22/comt](https://huggingface.co/datasets/shuai22/comt):
   - `TRAIN.jsonl`
   - `TEST.jsonl`
   - `comt.tar.gz` (image data)

2. Extract images and organize the directory as follows:
```text
CoVT-Reasoner/
├── data/
│   ├── TRAIN.jsonl
│   ├── TEST.jsonl
│   └── images_comt/      <-- Extracted from comt.tar.gz
│       ├── creation/
│       └── ... (other image subdirectories)
├── src/
├── transformers/
├── training_scripts/
└── README.md
```

3. **Data Format**: The dataset follows JSONL format with key fields:
   - `text_input`: Question/instruction
   - `image_input`: Initial input images
   - `sequence_plan`: Interleaved chain-of-thought rationale (text + helper_image paths)

### For RL Training
We use the ViRL39K dataset for RL training with Necessity Reward:

1. Download the dataset from [TIGER-Lab/ViRL39K](https://huggingface.co/datasets/TIGER-Lab/ViRL39K/)
2. The dataset contains visual reasoning tasks with ground truth answers (each sample generates 8 rollouts for group-wise reward computation)

## Training
### SFT Training (ILVR)
#### 1. Configure Training Script
Open `run_training.sh` and modify paths to match your local setup:

```bash
# In run_training.sh:
# Path to directory containing TRAIN.jsonl
DATA_PATH="/path/to/your/data" 

# Directory to save model checkpoints
SAVE_MODEL_PATH="/path/to/save/checkpoints"

# Training log file path
LOG_FILE="/path/to/save/train.log"

# (Optional) HuggingFace cache directory
export HF_HOME="/path/to/cache" 
```

#### 2. Start SFT Training
```bash
bash run_training.sh
```

**Default SFT Hyperparameters**:
- Base Model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Epochs: 15
- Gradient Accumulation Steps: 8
- Latent Size: 8

### RL Training with Necessity Reward
> [!NOTE]
> Recommended hardware for 7B model: **8 GPUs** (tested on 8x46G or 8x80G)

#### 1. Start RL Training
```bash
bash training_scripts/run_visionreasoner_7b_rl39k_allpenalty.sh
```

#### 2. Merge Checkpoint (Hugging Face Format)
```bash
python3 training_scripts/model_merger.py --local_dir [path_to_your_actor_checkpoint]
```
