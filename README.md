# Final Design Report
**SegLungAI** is a machine learning project focused on automating the detection and segmentation of neonatal lung anomalies in MRI scans. By leveraging advanced semantic segmentation techniques, the project aims to improve accuracy and reduce the need for manual intervention in medical imaging analysis.

## Table of Contents

1. [Team names and Project Abstract](#team-and-project-abstract)
2. [Project Description](#project-description)
3. [User Stories and Design Diagrams](#user-stories-and-design-diagrams)
   - [User Stories](#user-stories)
   - [Design Diagrams](#design-diagrams)
4. [Project Tasks and Timeline](#project-tasks-and-timeline)
   - [Task List](#task-list)
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

## Project Tasks and Timeline

### Task List
- Breakdown of tasks with roles assigned.

### Timeline
- Visual representation of the timeline.

### Effort Matrix
- Distribution of effort across team members.

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
