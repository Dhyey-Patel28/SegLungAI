# Project Milestones

We have identified following milestones for the **SegLungAI** neonatal lung segmentation project:

## 1. Data Augmentation and Preprocessing  
Preparing the limited neonatal dataset (initially n=13) for robust model training:
- Standardize NIfTI scans to RAS orientation.
- Resample to standardized voxel size (1 mm³).
- Resize and normalize slices to 256x256 pixels.
- Implement augmentation: rotations (±15°), translations (±5%), zooming (±15%), and horizontal flips.

## 2. Model Development and Training  
Implementing and training a supervised deep-learning segmentation model:
- Integrate **U-Net** architecture with a **ResNet-50** backbone pretrained on ImageNet.
- Implement a combined Dice and weighted Binary Cross Entropy (BCE) loss function.
- Configure training with Adam optimizer (learning rate=1e-4).
- Integrate training callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau.
- Save trained model checkpoints and training metrics (Dice coefficient, loss).

## 3. Post-Processing and Visualization  
Post-processing model outputs for accurate clinical interpretation:
- Threshold predictions to generate binary segmentation masks.
- Optionally apply morphological post-processing (closing operations).
- Create visual overlays (predicted vs. ground truth masks) for validation.
- Save generated masks in PNG, NIfTI, and MATLAB formats (.mat).

## 4. Evaluation and Metric Computation  
Assessing and documenting model performance rigorously:
- Compute Dice coefficient and Intersection over Union (IoU) per validation slice.
- Generate ROC curve (AUC metrics) to evaluate binary segmentation accuracy.
- Document metrics in visualizations: boxplots, ROC curves, confusion matrices.
- Ensure performance stability through thorough slice-level analysis.

## 5. Pipeline Generalization and Automation  
Creating a robust, flexible, and automated segmentation pipeline:
- Automate the pipeline end-to-end (data preprocessing → model inference → output storage).
- Generalize pipeline to support varied naming conventions and datasets.
- Save metadata systematically (slice number, patient ID, file paths) for traceability.

## 6. Presentation and Reporting  
Creating professional documentation and deliverables:
- Weekly update presentations to Dr. Jason Woods, Alex Matheson, and Abdullah Bdaiwi.
- Develop comprehensive documentation of the pipeline and methodology.
- Finalize project presentation and detailed final report for the senior design expo.

---

# Updated Project Timeline

| **Task**                                                | **Start Date**   | **End Date**     |
|---------------------------------------------------------|------------------|------------------|
| Research data augmentation techniques                   | Sep 15, 2024     | Sep 18, 2024     |
| Implement augmentation and preprocessing pipeline       | Sep 19, 2024     | Sep 25, 2024     |
| Develop model (U-Net + ResNet-50) integration           | Sep 26, 2024     | Oct 5, 2024      |
| Train model with augmented neonatal data                | Oct 6, 2024      | Oct 20, 2024     |
| Post-process and save predicted masks                   | Oct 21, 2024     | Oct 26, 2024     |
| Visualize overlays and prepare validation plots         | Oct 27, 2024     | Oct 30, 2024     |
| Evaluate model performance metrics (Dice, IoU, ROC)     | Oct 31, 2024     | Nov 3, 2024      |
| Generalize and automate the complete pipeline           | Nov 4, 2024      | Nov 10, 2024     |
| Weekly meetings and interim presentations               | Ongoing          | Ongoing          |
| Finalize documentation, presentation, and Expo prep     | Nov 11, 2024     | Nov 15, 2024     |

---

# Effort Matrix

| **Task**                                               | **Effort Score (1-5)** | **Milestone**                            |
|--------------------------------------------------------|------------------------|------------------------------------------|
| Research data augmentation and preprocessing           | 2                      | Data Augmentation and Preprocessing      |
| Implement augmentation and preprocessing pipeline      | 3                      | Data Augmentation and Preprocessing      |
| Integrate U-Net model with ResNet-50 backbone          | 5                      | Model Development and Training           |
| Train model with augmentation and callbacks            | 5                      | Model Development and Training           |
| Post-process and threshold predictions                 | 3                      | Post-Processing and Visualization        |
| Create visual overlays for validation                  | 3                      | Post-Processing and Visualization        |
| Compute Dice, IoU, ROC, and validation metrics         | 4                      | Evaluation and Metric Computation        |
| Generalize pipeline for flexible data handling         | 4                      | Pipeline Generalization and Automation   |
| Automate full pipeline workflow                        | 5                      | Pipeline Generalization and Automation   |
| Prepare weekly presentations and interim updates       | 2                      | Presentation and Reporting               |
| Finalize documentation and prepare Expo materials      | 4                      | Presentation and Reporting               |

---

### Total Effort
**Sum of Effort Scores:** 40

---

### Notes:
- **Effort Scoring System**:
  - **1**: Minimal effort (straightforward research or tasks).
  - **2**: Low effort (basic scripting, documentation).
  - **3**: Moderate effort (implementation of features or visualization).
  - **4**: High effort (metric evaluation, pipeline generalization).
  - **5**: Very high effort (complex integration, training, and full automation).
