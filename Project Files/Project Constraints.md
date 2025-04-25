# Project Constraints

In developing **SegLungAI**, a supervised deep-learning solution for automated neonatal lung segmentation in MRI scans, the following constraints are crucial for guiding the project's successful completion and practical utility:

**Economic** - 
The project primarily utilizes open-source software libraries such as TensorFlow, segmentation_models, and publicly accessible deep-learning frameworks due to limited funding. Computationally intensive model training requires leveraging institutional computational resources or cost-effective cloud platforms, imposing limitations on computational scalability and speed. Consequently, the size and complexity of the employed models (e.g., ResNet-50 backbone) must be carefully balanced against available computational resources.

**Professional** - 
The project's multidisciplinary nature demands professional standards adherence across both medical imaging and machine learning disciplines. Effective collaboration with clinical researchers—particularly advisor Dr. Jason Woods and collaborators Alex Matheson, PhD, and Abdullah Bdaiwi, PhD—necessitates clear communication, rigorous documentation, and transparent processes. Expertise in deep learning, semantic segmentation techniques, neonatal imaging, and clinical application requirements must be consistently maintained to ensure the project's accuracy, clinical relevance, and scientific integrity.

**Ethical** - 
Ethical considerations are paramount due to the sensitive nature of neonatal medical data. All MRI data used in this project must be fully anonymized, securely stored, and handled in strict accordance with ethical guidelines. Additionally, the model must undergo thorough validation to avoid any potential misdiagnosis or misinterpretation of clinical data, thereby safeguarding patient safety and outcomes. Adherence to ethical standards in medical AI, especially regarding accountability and transparency of automated predictions, is critical throughout the project lifecycle.

**Legal** - 
Compliance with legal standards, notably HIPAA regulations governing medical data privacy in the United States, is mandatory. All datasets must be securely anonymized, with explicit authorization for use in research settings. Furthermore, open-source tools, pretrained models, and third-party libraries utilized (e.g., segmentation_models and ResNet-50 pretrained weights) require careful attention to their respective licenses, ensuring proper attribution and adherence to intellectual property rights.

**Social** - 
The success and adoption of SegLungAI hinge on its practical integration into diverse clinical environments, ranging from resource-rich healthcare institutions to settings with limited infrastructure. This necessitates the development of a robust, intuitive user interface and seamless integration into existing clinical workflows, thus ensuring widespread accessibility and tangible benefit. The project aims to reduce clinical workload significantly and improve neonatal healthcare quality universally, underscoring the need for equitable, user-centered design principles.

By systematically addressing these economic, professional, ethical, legal, and social constraints, **SegLungAI** is positioned to deliver a clinically robust, ethically responsible, and socially impactful AI-driven solution for neonatal lung segmentation.
