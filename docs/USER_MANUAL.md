# SegLungAI — User Manual

Complete reference for the one-and-only script, all config options, and output formats.

---

## Table of Contents

1. [Installation](#installation)  
2. [Configuration Reference](#configuration-reference)  
3. [Command-Line Tool](#command-line-tool)  
4. [Pipeline Details](#pipeline-details)  
5. [Outputs & Formats](#outputs--formats)  
6. [Troubleshooting](#troubleshooting)  

---

## Installation

```bash
git clone https://github.com/Dhyey-Patel28/SegLungAI.git
cd SegLungAI
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Supported OS: Linux, macOS, Windows 10+ (Python 3.8+).

## Configuration Reference

All settings live in `config.py`:

| Parameter             | Description                                          | Default                                       |
|-----------------------|------------------------------------------------------|-----------------------------------------------|
| `DATA_PATH`           | Where to find your input volumes                    | `Neonatal Test Data - August 1 2024`          |
| `OUTPUT_BASE_DIR`     | Where to save runs                                   | `../LungSegmentation2D_Unet_BBResnet_Outputs` |
| `IMAGE_SIZE`          | 2D slice resolution (px)                             | `256`                                         |
| `BATCH_SIZE`          | Number of slices per training batch                  | `32`                                          |
| `TRAIN_TEST_SPLIT`    | Fraction of subjects held out for validation         | `0.15`                                        |
| `LEARNING_RATE`       | Initial learning rate                                | `1e-4`                                        |
| `NUM_EPOCHS`          | Number of training epochs                            | `1`                                           |
| `BACKBONE`            | Encoder architecture for U-Net                       | `resnet50`                                    |
| `THRESHOLD`           | Probability cutoff for binary mask                   | `0.1`                                         |
| `POSTPROCESS`         | Apply morphological closing & component filter       | `False`                                       |
| `SAVE_OVERLAY_IMAGES` | Save contour-overlay PNGs after inference            | `True`                                        |

---

## Command-Line Tool

### `Main_Resnet.py`

```bash
Usage: python Main_Resnet.py
```

No flags. To change behavior, edit `config.py`.

## Pipeline Details

1. **Preprocessing**
   - `reorient_to_standard` → RAS orientation
   - `resample_to_voxel_size` → target voxel spacing
  
2. **Data Generation**
   - `NiftiSliceSequence` slices volumes into 2D images
   - Optional augmentation via Keras `ImageDataGenerator`
  
3. **Model**
   - 2D U-Net (ResNet50 encoder, decoder filters (256,128,64,32,16))
   - Loss = Dice coefficient
  
4. **Inference & Post-processing**
   - Threshold at `THRESHOLD`
   - Optional morphological closing & keep two largest components

## Outputs & Formats

All saved under one run folder in `OUTPUT_BASE_DIR`:
- `masks/` – per-slice binary PNGs (with overlaid metadata if enabled)
- `all_comparisons/` – side-by-side ground-truth vs. prediction overlays
- `training_plots/` –
  - `training_validation_loss.png`
  - `validation_dice_boxplot.png`
  - `val_ROC_curve.png`
  - `val_confusion_matrix.png`
  - (and Hausdorff boxplot, if MedPy is installed)
- `mask_metadata.csv` – maps patient IDs and slice numbers to PNG filenames

## Troubleshooting

| Error                        | Cause                         | Fix                                          |
|------------------------------|-------------------------------|----------------------------------------------|
| `ModuleNotFoundError: cv2`   | OpenCV not installed          | `pip install opencv-python`                  |
| `CUDA out of memory`         | GPU memory insufficient       | Lower `BATCH_SIZE` in `config.py`            |
| `ValueError: shape mismatch` | Inconsistent slice dimensions | Ensure all volumes resampled to `IMAGE_SIZE` |

Still stuck? ▶️ Open a GitHub issue!
