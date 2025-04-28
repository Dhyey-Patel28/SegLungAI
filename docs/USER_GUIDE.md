# SegLungAI — User Guide

A friendly, step-by-step “how to” for first-time users. Jargon is minimized; where used, it’s explained.

---

## 1. What is SegLungAI?

SegLungAI automates lung segmentation in neonatal MRI scans, so that:

- **Radiologists** save hours of manual annotation  
- **Researchers** process large cohorts reproducibly  
- **Clinicians** visualize lung contours for anomaly screening  

*(See [User Stories](../Project%20Files/User%20Stories.md) for more)*

---

## 2. Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/Dhyey-Patel28/SegLungAI.git
   cd SegLungAI

2. **Create a Python 3.8+ virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # on Windows: venv\Scripts\activate

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt

## 3. Quick‐Start Usage

1. **Prepare your data**  
   - Create a folder of subject sub-directories, each containing your proton and mask volumes (e.g. `Proton.nii`, `Mask.nii`).  
   - Place that folder at the path defined by `DATA_PATH` in `config.py`.  
     By default, this is:
     ```
     <project_root>/Neonatal Test Data - August 1 2024
     ```

2. **Run the SegLungAI pipeline**  
   ```bash
   python Main_Resnet.py

3. **Inspect outputs in `results/`:**
   - `.nii.gz` masks
   - `.png` slice overlays
   - Metadata CSV
  
## 4. Typical Workflows

1. **Batch mode**
   ```bash
   python batch_process.py --config configs/batch_config.yaml

2. **View your results**
   By default they land in:
   ```bash
   LungSegmentation2D_Unet_BBResnet_Outputs/
     run_<YYYYMMDD_HHMMSS>/
       masks/             ← binary .png slices
       all_comparisons/   ← side-by-side overlays
       training_plots/    ← loss, dice, ROC, etc.
       mask_metadata.csv  ← patient+slice metadata

## 5. FAQ

**Q: What input formats are supported?**
A: NIfTI (.nii, .nii.gz) and single-slice DICOM (.dcm).

**Q: I get an “Out of memory” error**
- Lower `BATCH_SIZE` in `config.py`
- Or run on CPU by setting `gpu_id=-1` in the code (edit `Main_Resnet.py`)

**Q: How do I tweak preprocessing or augmentation?**
Edit `config.py` (e.g. `IMG_AUGMENTATION`, `TARGET_VOXEL_SIZE`, etc.)

**Q: Where do I report bugs?**
Open an [issue on GitHub](https://github.com/Dhyey-Patel28/SegLungAI/issues).
