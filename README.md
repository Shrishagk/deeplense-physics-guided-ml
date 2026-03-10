# DeepLense – ML4Sci GSoC Evaluation Tasks

This repository contains my solutions for the evaluation tasks of the **ML4Sci DeepLense project** for **Google Summer of Code (GSoC)**.

## Project Overview
The goal is to classify gravitational lensing images using deep learning models. The dataset contains three classes:
- Strong lensing images with **no substructure**
- **Subhalo** substructure
- **Vortex** substructure

The models are implemented using **PyTorch** and evaluated using standard metrics such as **ROC curves** and **AUC scores**.

## Implemented Models
The following architectures were implemented and compared:

| Model | Validation Accuracy |
|------|--------------------|
| SimpleCNN | 89.15% |
| ImprovedCNN | 92.87% |
| ResNet-18 | 95.17% |

## Evaluation
- Train/Test Split: **90:10**
- Metrics used:
  - Accuracy
  - ROC Curve
  - AUC Score



## Next Steps
The next stage involves developing a **Physics-Informed Neural Network (PINN)** that incorporates the **gravitational lensing equation** to improve classification performance.

## Author
Shrisha G K  
GSoC Applicant – ML4Sci DeepLense Project
