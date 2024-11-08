# **Project Description**

## **Team Name**
SegLungAI (TBD)

## **Team Member(s)**

- **Dhyey Patel**  
  Major: Computer Science  
  Email: [patel4du@mail.uc.edu](mailto:patel4du@mail.uc.edu)

## **Project Topic Area**
The **SegLungAI** project aims to develop an AI-driven solution for automatic lung anomaly detection and segmentation in neonates using CT and MRI chest scans. The primary objective is to achieve high-accuracy lung segmentation through semantic segmentation techniques in Python. The project will begin by segmenting lungs from chest scans of neonates, utilizing 30 anonymized images provided by Cincinnati Children's Hospital. As the project progresses, new goals and features will be added to further enhance model performance.

### **Faculty Advisor**
- **Jason C. Woods, PhD**  
  Professor, UC Department of Pediatrics  
  Email: [jason.woods@cchmc.org](mailto:jason.woods@cchmc.org)  
  Dr. Woods will serve as the project supervisor, providing guidance, task assignment, and support throughout the project.

### **Additional Guidance**
- **Alex Mathewson, PhD**  
  Research Fellow, Department of Pulmonary Medicine  
  Email: [alexander.matheson@cchmc.org](mailto:alexander.matheson@cchmc.org)  
  Dr. Mathewson will assist by providing resources, data, and any existing codebase, offering guidance as needed throughout the project.

## **Project Abstract**
**SegLungAI** will leverage semantic segmentation techniques to automatically detect and segment lung regions in chest CT and MRI scans. The goal is to develop an AI model in Python that achieves high accuracy in lung segmentation, reducing the need for manual corrections. As the project progresses, additional features and improvements will be incorporated to enhance model performance and extend the project’s capabilities.

## **Problem Statement**
Previously, lung segmentation from chest CT/MRI scans was done manually, taking months to complete for each patient and scan. While there are existing tools for segmentation, they still lack the accuracy needed to eliminate the need for manual review, which still requires human oversight for correcting mislabels or missing data.

## **Current Solutions and Gaps**
Current neonatal lung segmentation methods often lack the precision required for clinical use. Existing tools fail to fully address the complexities of neonatal CT/MRI scans, which vary in quality and resolution, leading to inaccuracies that require human intervention.

## **Technical Background**
The project will utilize Python and machine learning, focusing on semantic segmentation techniques with libraries such as TensorFlow or PyTorch. Medical imaging tools like OpenCV will also be employed. A deep learning model will be trained on 30 anonymized neonatal chest CT/MRI scans provided by Cincinnati Children’s Hospital.

## **Approach**
The project will begin with the pre-processing of the provided image data. A semantic segmentation model will be developed to detect and segment lung structures. The model's performance will be evaluated and iteratively improved to achieve higher accuracy and minimize the need for manual review.

## **Lung Segmentation Example**

An example of lung segmentation can be seen in the following case, where bronchopulmonary segments are annotated on an arterial phase axial chest CT scan:

**Case courtesy of Peter Jenvey, [Radiopaedia.org](https://radiopaedia.org/?lang=us). From the case [rID: 54511](https://radiopaedia.org/cases/bronchopulmonary-segments-annotated-ct-2).**

<img src="./assets/axial-segment.png" width="400"/>

Annotated bronchopulmonary segments on arterial phase axial chest CT.

This image highlights the detailed segmentation of lung regions, which is similar to the goal of our project—accurately segmenting neonate lung images using AI-driven methods.

## **Goals**
1. Segment lungs from neonatal chest CT/MRI scans.
2. Continuously refine the segmentation model to enhance accuracy and reduce manual oversight.
3. Expand the project to include additional datasets and improve the model’s performance across a wider range of cases.
