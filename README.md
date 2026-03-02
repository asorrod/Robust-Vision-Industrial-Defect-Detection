# Robust-Vision-Industrial-Defect-Detection
Robust Vision is an ongoing computer vision project focused on industrial surface defect detection using deep learning.

The project is built around the NEU Surface Defect Database, a benchmark dataset for steel surface inspection, and explores robustness and generalization challenges common in real industrial environments.

##🎯 Objective

To develop a reliable and robust defect classification system capable of identifying different types of surface defects under varying conditions such as:

-Illumination changes

-Noise perturbations

-Limited dataset size

-Class imbalance

The project emphasizes robustness and industrial applicability over benchmark-only optimization.

##🏭 Industrial Context

In industrial production lines, automated visual inspection systems must:

-Detect defects in real-time

-Maintain high precision and recall

-Generalize across changing environmental conditions

-Operate reliably with limited labeled data

This project simulates these challenges and evaluates model performance accordingly.

##🧠 Model Architecture

-Backbone: ResNet18

-Transfer Learning with fine-tuning

-Data augmentation strategies to improve generalization

-Cross-entropy loss for multi-class classification

-The choice of ResNet18 balances:

-Performance

-Computational efficiency

-Deployability in industrial environments

##📊 Dataset

NEU Surface Defect Database

The dataset contains 6 types of surface defects in hot-rolled steel strips:

-Crazing (Cr)

-Inclusion (In)

-Patches (Pa)

-Pitted Surface (PS)

-Rolled-in Scale (RS)

-Scratches (Sc)

Images are grayscale and represent real industrial surface inspection scenarios.

##📈 Evaluation

The system is evaluated using:

-Accuracy

-Precision

-Recall

-Confusion Matrix

-Robustness tests under synthetic noise / perturbations

##🛠 Tech Stack

-Python

-PyTorch

-Torchvision

-NumPy

-Matplotlib

-OpenCV

##🚀 Future Improvements

-Domain adaptation techniques

-Anomaly detection approaches (one-class methods)

-Data-efficient training strategies

-Lightweight deployment for edge devices

-Real-time inference optimization

##🧪 Project Status

Currently under active development.
Focus areas: robustness experiments and model generalization improvements.
