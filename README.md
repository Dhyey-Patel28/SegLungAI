# Final Design Report
**SegLungAI** is a machine learning project focused on automating the detection and segmentation of neonatal lung anomalies in MRI scans. By leveraging advanced semantic segmentation techniques, the project aims to improve accuracy and reduce the need for manual intervention in medical imaging analysis.

## Table of Contents

1. [Team names and Project Abstract](#team-and-project-abstract)
2. [Project Description](#project-description)
3. [User Stories and Design Diagrams](#user-stories-and-design-diagrams)
   - [User Stories](#user-stories)
   - [Design Diagrams](#design-diagrams)
4. [Project Milestones, Timeline and Effort Matrix](#project-milestones-timeline-and-effort-matrix)
   - [Project Milestones](#project-milestones)
   - [Timeline](#timeline)
   - [Effort Matrix](#effort-matrix)
5. [ABET Concerns Essay](#abet-concerns-essay)
6. [PPT Slideshow](#ppt-slideshow)
7. [Self-Assessment Essays](#self-assessment-essays)
8. [Professional Biographies](#professional-biographies)
9. [Budget](#budget)
10. [Appendix](#appendix)

---

## Team and Project Abstract

### Team
- **Jason C. Woods, PhD** (Faculty Advisor)
- **Alex Matheson, PhD** (Research Fellow)
- **Abdullah Bdaiwi, PhD** (Research Fellow)

### Project Abstract
SegLungAI focuses on developing a machine learning (ML)-based solution to automatically detect and segment lung regions in neonatal chest MRI scans. The project aims to achieve high segmentation accuracy using semantic segmentation techniques, with the goal of reducing manual corrections. Over time, the project will evolve to enhance model performance and adapt to more diverse datasets.

---

## Project Description

### Team Members
- **Jason C. Woods, PhD** (Faculty Advisor)  
  Professor, UC Department of Pediatrics  
  Email: [jason.woods@cchmc.org](mailto:jason.woods@cchmc.org)

- **Alex Matheson, PhD** (Advisor)  
  Research Fellow, Department of Pulmonary Medicine within the CPIR  
  Email: [alexander.matheson@cchmc.org](mailto:alexander.matheson@cchmc.org)

- **Abdullah Bdaiwi, PhD** (Advisor)  
  Research Fellow, Department of Pulmonary Medicine within the CPIR  
  Email: [abdullah.bdaiwi@cchmc.org](mailto:abdullah.bdaiwi@cchmc.org)

- **Dhyey Patel**  
  Major: Computer Science  
  Email: [patel4du@mail.uc.edu](mailto:patel4du@mail.uc.edu)

---

### Project Topic Area
The **SegLungAI** project aims to develop an AI-driven solution for automatic lung anomaly detection and segmentation in neonates using CT and MRI chest scans. The primary objective is to achieve high-accuracy lung segmentation through semantic segmentation techniques in Python. The project will begin by segmenting lungs from chest scans of neonates, utilizing 30 anonymized images provided by Cincinnati Children's Hospital. Over time, additional goals and features will be introduced to further enhance model performance.

---

### Project Abstract
**SegLungAI** leverages semantic segmentation techniques to automatically detect and segment lung regions in neonatal chest CT and MRI scans. The project uses a U-Net model with a ResNet-34 backbone to ensure high accuracy while addressing the unique challenges of neonatal imaging. With a focus on scalability, the project utilizes open-source tools and resources provided by Cincinnati Children's Hospital. Collaborating with medical professionals, SegLungAI aims to streamline diagnostic workflows and enhance precision in neonatal healthcare.

---

### Problem Statement
Manual lung segmentation from neonatal CT/MRI scans is time-consuming, often taking months to complete for each patient. Current tools lack the precision needed to eliminate manual review, requiring human oversight for correcting mislabels or missing data. This project addresses these challenges by automating the process to save time and improve accuracy.

---

### Current Solutions and Gaps
Existing neonatal lung segmentation methods often fail to fully address the complexities of neonatal scans, such as variations in quality and resolution. This results in inaccuracies that require human intervention, highlighting the need for a robust and automated solution like **SegLungAI**.

---

### Technical Background
The project utilizes Python and machine learning frameworks like TensorFlow and PyTorch, focusing on semantic segmentation techniques. Tools such as OpenCV will be used for image processing. The model will be trained on 30 anonymized neonatal chest CT/MRI scans provided by Cincinnati Children’s Hospital.

---

### Approach
1. **Data Preprocessing**: Initial image pre-processing includes resizing, normalization, and augmentation techniques to improve model performance.
2. **Model Development**: Implementing a U-Net model with a ResNet-34 backbone to achieve precise lung segmentation.
3. **Evaluation**: Iteratively improving the model using metrics like Dice Similarity Coefficient (DSC) and Intersection over Union (IoU).
4. **Automation**: Developing a pipeline to generalize segmentation capabilities across various neonatal imaging datasets.

---

### Goals
1. **Segment Neonatal Lungs from MRI/CT Scans**  
   - Create a robust pipeline for lung segmentation using semantic segmentation techniques.  
   - Achieve high accuracy and reduce manual corrections.  

2. **Refine and Optimize the Model**  
   - Evaluate model performance using standard metrics like IoU and DSC.  
   - Implement data augmentation and active learning for improved accuracy and generalizability.  

3. **Expand and Generalize**  
   - Incorporate additional datasets and imaging modalities, such as X-ray and ultrasound.  
   - Utilize transfer learning to enhance initial performance.  
   - Adapt the solution for broader pediatric imaging tasks beyond lung segmentation.

---

## User Stories and Design Diagrams

### User Stories
1. **Radiologist**
   - **As a radiologist**, I want a tool that automates lung segmentation in neonatal MRI scans **so that** I can reduce time spent on manual annotations.

2. **Medical Researcher**
   - **As a medical researcher**, I want a reliable and reproducible AI pipeline **so that** I can analyze large datasets of neonatal chest images more efficiently.

3. **Neonatal Clinician**
   - **As a neonatal clinician**, I want accurate lung segmentation outputs **so that** I can assist in identifying anomalies and ensure better patient outcomes.

### Design Diagrams

**SegLungAI** includes three levels of design diagrams that outline the system's functionality and structure:

#### D0 Diagram
- Represents the highest-level overview of the system.
- Highlights basic inputs (MRI scans) and outputs (segmented lung masks).
![alt text](./Design%20Diagrams/Design_Diagram_D0.png)

#### D1 Diagram
- Breaks the system into major subsystems:
  - Data Input and Preprocessing
  - Semantic Segmentation Model
  - Post-Processing
  - Evaluation Metrics
![alt text](./Design%20Diagrams/Design_Diagram_D1.png)

#### D2 Diagram
- Provides detailed breakdowns of the subsystems:
  - Preprocessing components such as resizing and augmentation.
  - Model training with specific layers and configurations.
  - Post-processing steps like thresholding and visualization.
  - Evaluation methods, including Dice Similarity Coefficient and Intersection over Union metrics.
![alt text](./Design%20Diagrams/Design_Diagram_D2.png)

#### Diagram Interpretation
- **Symbols and Conventions**:
  - **Rectangles**: Processes or modules.
  - **Parallelograms**: Data inputs/outputs.
  - **Ovals**: Start/end points of workflows.
  - **Arrows**: Flow of data and interactions between modules.
  - **Solid lines**: Direct data flows or processes.
  - **Dotted lines**: Optional or secondary interactions to avoid overlapping paths.

---

## Project Milestones, Timeline, and Effort Matrix

### Project Milestones

We have identified the following major milestones for the **SegLungAI** project:

#### 1. Data Augmentation and Preprocessing
This milestone involves preparing the dataset for training by applying augmentation techniques to improve model generalizability. Tasks include:
- Resizing images and masks.
- Implementing augmentation techniques like rotation, zooming, and flipping.
- Ensuring all preprocessing steps are integrated into a reproducible pipeline.

#### 2. Model Development and Training
This milestone focuses on implementing and training the semantic segmentation model:
- Modifying Abood's code to fit the project goals.
- Integrating the ResNet-34 backbone with the U-Net architecture.
- Training the model using augmented data and saving intermediate results.

#### 3. Post-Processing and Visualization
This milestone involves processing model outputs and preparing visualizations:
- Thresholding model predictions to generate binary masks.
- Overlaying predictions on input images for validation.
- Saving outputs in various formats (e.g., PNG, NIfTI, MAT).

#### 4. Evaluation and Metric Computation
This milestone emphasizes the evaluation of the model's performance:
- Implementing IoU and Dice Coefficient calculations.
- Generating training/validation plots for loss and IoU metrics.
- Documenting results for comparison across experiments.

#### 5. Generalization and Automation
This milestone focuses on making the pipeline user-friendly and reusable:
- Generalizing the model for various file naming conventions.
- Automating the workflow for seamless operation from data input to output.

#### 6. Presentation and Reporting
This milestone involves creating deliverables for meetings and the final report:
- Preparing weekly presentations for discussions with Dr. Jason Woods, Alex, and Abood.
- Documenting the complete pipeline and project findings in a final report.

---

### Timeline

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

### Effort Matrix

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

---

## ABET Concerns Essay
- Address ethical, global, economic, and societal concerns.

---

## PPT Slideshow
- Include visuals and insights covering the project, tasks, and ABET concerns.

---

## Self-Assessment Essays
- Reflections from each team member.

---

## Professional Biographies
- Team member backgrounds and expertise.

---

## Budget
- Expenses and monetary value of donated items with sources listed.

---

## Appendix
- References, citations, links to code repositories, meeting notes, and evidence of 45 hours of effort per team member.
