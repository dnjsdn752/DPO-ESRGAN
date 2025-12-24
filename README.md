# DPO-ESRGAN

**Direct Preference Optimization for Super-Resolution (DPO-ESRGAN)**

[![Paper](https://img.shields.io/badge/Paper-Electronics-blue)](https://doi.org/10.3390/electronics14173357)
[![DOI](https://img.shields.io/badge/DOI-10.3390%2Felectronics14173357-blue)](https://doi.org/10.3390/electronics14173357)

This repository contains the official implementation of the paper:
**"Applying Direct Preference Optimization to Super-Resolution"**  
*Electronics 2025, 14(17), 3357*  
[https://doi.org/10.3390/electronics14173357](https://doi.org/10.3390/electronics14173357)

## Abstract
Super-resolution (SR) is a long-standing task in the field of computer vision that aims to improve the quality and resolution of an image. ESRGAN is a representative generative adversarial network specialized to produce perceptually-convincing SR images. However, it often fails to recover local details and still produces blurry or unnatural visual artifacts, resulting in producing SR images that people do not prefer. To address this problem, we propose to adopt direct preference optimization (DPO), originally devised to fine-tune large language models based on human preferences. To this end, we develop a method for applying DPO to ESRGAN, and add a DPO loss for training the ESRGAN generator. Through ×4 SR experiments utilizing benchmark datasets, it is demonstrated that the proposed method can produce SR images with significantly higher perceptual quality and higher human preference than ESRGAN and other ESRGAN variants that modified the loss or network structure of ESRGAN. Specifically, when compared to ESRGAN, the proposed method achieved 0.27 lower PieAPP values, 0.34 lower NIQE values, and 0.17 lower PSNR values on average.

**Keywords**: Image super-resolution, ESRGAN, direct preference optimization, PieAPP, LPIPS

> **Note**: This code is derived from [Pie-ESRGAN-PyTorch](https://github.com/cyun-404/PieESRGAN) and adapted for DPO experiments.

---

## 📋 Table of Contents
- [Prerequisites](#prerequisites-)
- [Dataset Preparation](#dataset-preparation-)
- [Training](#training-)
- [Testing & Evaluation](#testing--evaluation-)
- [Citation](#citation-)
- [License](#license-)
- [Acknowledgements](#acknowledgements-)

---

## Prerequisites 🛠

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

## Dataset Preparation 📂

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

## Training 🚀

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

## Testing & Evaluation 📊

We evaluate the performance using several metrics. The arrows indicate whether higher or lower values are better:
- **PSNR** ↑ (Peak Signal-to-Noise Ratio)
- **SSIM** ↑ (Structural Similarity)
- **LPIPS** ↓ (Learned Perceptual Image Patch Similarity)
- **PieAPP** ↓ (Perceptual Image-Error Assessment through Pairwise Preference)
- **NIQE** ↓ (Naturalness Image Quality Evaluator)

### Quick Test
**`test.py`**: A simple script to evaluate performance on a single dataset.
```bash
# Evaluate on default dataset (Urban100) or specify one
python test.py --dataset_name Set5
```

### Detailed Evaluation Scripts
The `test/` directory contains specialized scripts for various evaluation needs:

- **`test/test_all.py`**: Runs a comprehensive benchmark on all major datasets (Set5, Set14, BSD100, Urban100) to measure overall performance.
- **`test/test_on_image.py`**: Performs inference and metric calculation on a **single specific image file** to visualize results.
  ```bash
  python test/test_on_image.py --image_path "data/Set5/baby.png"
  ```
- **`test/test_img.py`**: Calculates metrics (PSNR, SSIM, LPIPS, Pie) for all images in a target folder.
- **`test/test_ssim.py`**: Specialized script specifically for calculating **SSIM** scores.
- **`test/test_niqe.py`**: Specialized script specifically for calculating **NIQE** scores.

---

## Citation 📖

If you find this project useful for your research, please cite our paper:

```bibtex
@article{electronics14173357,
  title={Applying Direct Preference Optimization to Super-Resolution},
  journal={Electronics},
  volume={14},
  year={2025},
  number={17},
  article-number={3357},
  url={https://www.mdpi.com/2079-9292/14/17/3357},
  issn={2079-9292},
  doi={10.3390/electronics14173357}
}
```

---

## License 📝

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## Acknowledgements 🙌

This code is built upon the excellent work of [Pie-ESRGAN-PyTorch](https://github.com/cyun-404/PieESRGAN). We thank the authors for their open-source contribution.
