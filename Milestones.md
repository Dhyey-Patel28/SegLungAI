# Project Milestones

We have identified the following major milestones for our project:

## 1. Data Augmentation and Preprocessing  
This milestone involves preparing the dataset for training by applying augmentation techniques to improve model generalizability. Tasks include:
- Resizing images and masks.
- Implementing augmentation techniques like rotation, zooming, and flipping.
- Ensuring all preprocessing steps are integrated into a reproducible pipeline.

## 2. Model Development and Training  
This milestone focuses on implementing and training the semantic segmentation model:
- Modifying Abood's code to fit the project goals.
- Integrating the ResNet-34 backbone with the U-Net architecture.
- Training the model using augmented data and saving intermediate results.

## 3. Post-Processing and Visualization  
This milestone involves processing model outputs and preparing visualizations:
- Thresholding model predictions to generate binary masks.
- Overlaying predictions on input images for validation.
- Saving outputs in various formats (e.g., PNG, NIfTI, MAT).

## 4. Evaluation and Metric Computation  
This milestone emphasizes the evaluation of the model's performance:
- Implementing IoU and Dice Coefficient calculations.
- Generating training/validation plots for loss and IoU metrics.
- Documenting results for comparison across experiments.

## 5. Generalization and Automation  
This milestone focuses on making the pipeline user-friendly and reusable:
- Generalizing the model for various file naming conventions.
- Automating the workflow for seamless operation from data input to output.

## 6. Presentation and Reporting  
This milestone involves creating deliverables for meetings and the final report:
- Preparing weekly presentations for discussions with Dr. Jason Woods, Alex, and Abood.
- Documenting the complete pipeline and project findings in a final report.

---

# Timeline

| **Task**                                               | **Start Date**   | **End Date**     |
|--------------------------------------------------------|------------------|------------------|
| Research data augmentation techniques                 | Sep 15, 2024     | Sep 18, 2024     |
| Implement data augmentation pipeline                  | Sep 19, 2024     | Sep 22, 2024     |
| Preprocess dataset (resize, normalize)                | Sep 23, 2024     | Sep 25, 2024     |
| Modify Abood's code for model integration             | Sep 26, 2024     | Sep 30, 2024     |
| Train U-Net with ResNet-34 backbone                   | Oct 1, 2024      | Oct 10, 2024     |
| Save and document training outputs (metrics/plots)    | Oct 11, 2024     | Oct 13, 2024     |
| Generate and save binary masks                        | Oct 14, 2024     | Oct 16, 2024     |
| Overlay masks on images for validation                | Oct 17, 2024     | Oct 19, 2024     |
| Compute IoU and Dice Coefficient                      | Oct 20, 2024     | Oct 23, 2024     |
| Generalize pipeline for flexible input formats        | Oct 24, 2024     | Oct 27, 2024     |
| Automate pipeline from input to output                | Oct 28, 2024     | Oct 31, 2024     |
| Prepare weekly presentations                          | Ongoing          | Ongoing          |
| Finalize project documentation and report             | Nov 1, 2024      | Nov 5, 2024      |

---

# Effort Matrix

| **Task**                                               | **Effort Score (1-5)** | **Milestone**                          |
|--------------------------------------------------------|------------------------|----------------------------------------|
| Research data augmentation techniques                 | 2                      | Data Augmentation and Preprocessing    |
| Implement data augmentation pipeline                  | 3                      | Data Augmentation and Preprocessing    |
| Preprocess dataset (resize, normalize)                | 2                      | Data Augmentation and Preprocessing    |
| Modify Abood's code for model integration             | 5                      | Model Development and Training         |
| Train U-Net with ResNet-34 backbone                   | 5                      | Model Development and Training         |
| Save and document training outputs (metrics/plots)    | 2                      | Post-Processing and Visualization      |
| Generate and save binary masks                        | 3                      | Post-Processing and Visualization      |
| Overlay masks on images for validation                | 3                      | Post-Processing and Visualization      |
| Compute IoU and Dice Coefficient                      | 4                      | Evaluation and Metric Computation      |
| Generalize pipeline for flexible input formats        | 4                      | Generalization and Automation          |
| Automate pipeline from input to output                | 5                      | Generalization and Automation          |
| Prepare weekly presentations                          | 2                      | Presentation and Reporting             |
| Finalize project documentation and report             | 4                      | Presentation and Reporting             |

---

### Total Effort
**Sum of Effort Scores:** 44

---

### Notes:
- **Effort Scoring System**:
  - **1**: Low effort (straightforward research).
  - **2**: Slightly more effort (basic implementation or documentation).
  - **3**: Moderate effort (UI/visualization design or coding).
  - **4**: High effort (evaluation metrics or data generalization).
  - **5**: Very high effort (complex backend or model development).
