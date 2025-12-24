# DPO-ESRGAN

**Direct Preference Optimization for Super-Resolution (DPO-ESRGAN)**

This repository contains the implementation of **DPO-ESRGAN**, a research project applying Direct Preference Optimization (DPO) to Single Image Super-Resolution (SISR). By leveraging preference data, this model aims to generate high-fidelity super-resolved images that align better with human perceptual preferences.

> **Note**: This code is derived from [Pie-ESRGAN-PyTorch](https://github.com/cyun-404/PieESRGAN) and adapted for DPO experiments.

---

## 📋 Table of Contents
- [Prerequisites](#prerequisites)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Testing & Evaluation](#testing--evaluation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## 🛠 Prerequisites

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/dnjsdn752/DPO-ESRGAN.git
   cd DPO-ESRGAN
   ```

2. Install python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📂 Dataset Preparation

### Directory Structure
Place your High-Resolution (HR) training and validation datasets in the `data` directory.
The default configuration expects **DIV2K** dataset structure as follows:

```text
data/
  ├── DIV2K_train_HR/      # Training High-Resolution images
  └── DIV2K_valid_HR/      # Validation High-Resolution images
```

> You can modify the dataset paths in `config.py` if needed.

---

## 🚀 Training

Training DPO-ESRGAN consists of two main stages. Since ESRGAN-based models are difficult to train from scratch, we follow a pre-training strategy.

### Stage 1: Generator Pre-training
First, train the standard Generator (without DPO or Adversarial loss) to initialize the weights.

1. **Configure `config.py`**:
   - Set `start_p_epoch = 0` to start pre-training from scratch.
   - Set `p_epochs` to a sufficient number (e.g., 2000) for convergence.
   - Set `start_epoch` and `epochs` to 0 or ignore them for this stage.

2. **Run Training**:
   ```bash
   python SR_DPO_train.py
   ```
   *(This will primarily train the PSNR-oriented generator first)*

3. **Save Weights**:
   - After pre-training, locate the best model weight file (e.g., `p-best.pth` or similar in `results/`).
   - Rename/move it to: `results/pieonly/g-best.pth`
   - This file will be used as the starting point (Reference Model) for the DPO training stage.

### Stage 2: DPO Training
Once the pre-trained weights (`g-best.pth`) are placed in `results/pieonly/`, you can proceed with the main DPO training. Choose one of the following scripts based on your experimental needs:

#### 1. Standard DPO Training (`SR_DPO_train.py`)
This is the **standard implementation** of the DPO method as described in the research paper. It uses the reference model to guide the preference optimization.
```bash
python SR_DPO_train.py
```

#### 2. No-Reference DPO Training (`SR_DPO_NoRef_train.py`)
This script implements a **Reference-Free** variation of the DPO training. It removes the dependency on the pre-trained reference model during the preference loss calculation.
```bash
python SR_DPO_NoRef_train.py
```

#### 3. DPO-Only Training (`SR_DPO_Only_train.py`)
This is an experimental baseline where the DPO loss is simply attached to the SR model without the full DPO framework logic. 
> **Note**: Performance may be lower than the standard method; this is mainly for experimental comparison.
```bash
python SR_DPO_Only_train.py
```

---

## 📊 Testing & Evaluation

Evaluate the trained models using the scripts provided in the `test/` directory.

### Benchmark Evaluation
Run a comprehensive test across multiple datasets (Set5, Set14, BSD100, Urban100) to calculate PSNR, SSIM, LPIPS, and PieAPP scores.
```bash
python test/test_all.py
```

### Single Image Inference
Test the model on a specific image to visualize the super-resolution result.
```bash
python test/test_on_image.py --image_path "data/Set5/baby.png"
```

### Metric Calculation
- **`test/test_ssim.py`**: Measure SSIM for a dataset.
- **`test/test_niqe.py`**: Measure NIQE scores.
- **`test/test_img.py`**: Comprehensive metric calculation for a folder of images.

---

## 📝 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## 🙌 Acknowledgements

This code is built upon the excellent work of [Pie-ESRGAN-PyTorch](https://github.com/cyun-404/PieESRGAN). We thank the authors for their open-source contribution.
