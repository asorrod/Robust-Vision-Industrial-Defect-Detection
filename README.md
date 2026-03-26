# Robust-Vision-Industrial-Defect-Detection
Robust Vision is a computer vision project focused on industrial defect detection under real-world conditions.

Unlike standard approaches optimized for clean datasets, this project emphasizes robustness and generalization by evaluating model performance under multiple types of visual corruption (noise, blur, brightness changes, and occlusions).

The goal is to simulate real industrial environments where data is imperfect and conditions are constantly changing.

## 🎯 Objective

To develop a reliable and robust defect classification system capable of identifying different types of surface defects under varying conditions such as:

- Illumination changes

- Noise perturbations

- Limited dataset size

- Class imbalance

The project emphasizes robustness and industrial applicability over benchmark-only optimization.

## 🏭 Industrial Context

In industrial production lines, automated visual inspection systems must:

- Detect defects in real-time

- Maintain high precision and recall

- Generalize across changing environmental conditions

- Operate reliably with limited labeled data

This project simulates these challenges and evaluates model performance accordingly.

## 🧠 Model Architecture

- Backbone: ResNet18

- Transfer Learning with fine-tuning

- Data augmentation strategies to improve generalization

- Cross-entropy loss for multi-class classification

- The choice of ResNet18 balances:

- Performance

- Computational efficiency

- Deployability in industrial environments

## 📊 Dataset

NEU Surface Defect Database

The dataset contains 6 types of surface defects in hot-rolled steel strips:

- Crazing (Cr)

- Inclusion (In)

- Patches (Pa)

- Pitted Surface (PS)

- Rolled-in Scale (RS)

- Scratches (Sc)

Images are grayscale and represent real industrial surface inspection scenarios.

## 📈 Evaluation

The system is evaluated using:

- Accuracy

- Precision

- Recall

- Confusion Matrix

- Robustness tests under synthetic noise / perturbations

## 🛠 Tech Stack

- Python

- PyTorch

- Torchvision

- NumPy

- Matplotlib

- OpenCV

## 🚀 Future Improvements

- Domain adaptation techniques

- Anomaly detection approaches (one-class methods)

- Data-efficient training strategies

- Lightweight deployment for edge devices

- Real-time inference optimization

## ⚙️ Experimental Setup

- Corruptions applied:
  - Gaussian noise
  - Gaussian blur
  - Brightness shifts
  - Occlusions
- Severity levels: 1 (low) to 5 (high)
- Evaluation performed across all corruption combinations

## 📊 Results

Model Comparison

| Model | Model	Mean F1 (Severity=1)	| Mean F1 (Severity=3)	| Mean F1 (Severity=5) |
| ----- | --------------------------- | --------------------- | ---------------- |
| Baseline |	0.607137 |	0.371032 | 0.314033 |
| Robust |	0.926033 | 0.801130 | 0.712048 |

The robust model maintains significantly higher performance under severe corruptions, demonstrating improved generalization and stability.

## 📉 Visual Results

<img width="3182" height="1638" alt="robust_heatmap" src="https://github.com/user-attachments/assets/d4b4e667-9325-4f8b-8ec0-b153d8c0857d" />

<img width="1702" height="2154" alt="robustness_under_severe_corruptions" src="https://github.com/user-attachments/assets/8b3c8979-047b-4d75-884b-ace8c109f384" />

