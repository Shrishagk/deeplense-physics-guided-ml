# DeepLense – ML4Sci GSoC Evaluation Tasks

This repository contains my solutions for the evaluation tasks of the **ML4Sci DeepLense project** for **Google Summer of Code (GSoC)**.

---

## 🔷 Project Overview

The objective of this project is to classify **strong gravitational lensing images** using deep learning techniques.

The dataset consists of three classes:

* Strong lensing images with **no substructure**
* **Subhalo (sphere)** substructure
* **Vortex** substructure

The models are implemented using **PyTorch** and evaluated using robust metrics such as **ROC curves** and **AUC scores**.

---

## 🔷 Common Test – Baseline Models

To establish a strong baseline, multiple CNN architectures were implemented and evaluated:

| Model       | Validation Accuracy |
| ----------- | ------------------- |
| SimpleCNN   | 89.15%              |
| ImprovedCNN | 92.87%              |
| ResNet-18   | 95.17%              |

📌 **Insight:**
Performance improves with architectural complexity; however, these models rely purely on data-driven learning and do not incorporate physical constraints.

---

## 🔷 Specific Test – Physics-Informed Neural Network (PINN)

To address the limitations of standard CNNs, a **Physics-Informed Neural Network (PINN)** was developed by integrating **gravitational lensing physics** into the training process.

### ✅ Key Features

* CNN backbone with multi-task outputs
* Physics-based loss incorporating lensing constraints
* Lambda annealing for stable training
* Uncertainty estimation using Monte Carlo Dropout

---

## 🔷 PINN Performance

| Metric              | Value      |
| ------------------- | ---------- |
| Accuracy            | **95.33%** |
| Macro ROC-AUC       | **0.9934** |
| MC Dropout Accuracy | **95.33%** |
| Mean Entropy        | **0.2449** |

📌 **Key Observation:**
The PINN achieves performance comparable to ResNet-18 while additionally ensuring **physical consistency and improved interpretability**.

---

## 🔷 Class-wise Performance

| Class   | Precision | Recall | F1-score |
| ------- | --------- | ------ | -------- |
| No Lens | 0.9183    | 0.9976 | 0.9563   |
| Sphere  | 0.9771    | 0.9036 | 0.9389   |
| Vortex  | 0.9697    | 0.9588 | 0.9642   |

📌 **Insight:**

* High recall for *no-lens* class indicates strong detection capability
* Slight confusion between *sphere* and *vortex* reflects structural similarity

---

## 🔷 Evaluation Metrics

* Train/Test Split: **90:10**
* Metrics used:

  * Accuracy
  * ROC Curve
  * AUC Score
  * Confusion Matrix
  * Uncertainty (MC Dropout, Entropy)

---

## 🔷 Key Contributions

* Implemented and compared multiple CNN architectures
* Designed a **Physics-Informed Neural Network for classification**
* Integrated **physics-based loss functions**
* Applied **uncertainty estimation techniques**
* Built a **complete end-to-end training and evaluation pipeline**

---

## 🔷 Results Summary

* Achieved **>95% accuracy** across models
* Developed a **novel PINN-based classification approach**
* Demonstrated that **physics-informed learning improves reliability without sacrificing performance**

---

## 🔷 Future Work

* Extend the model to **real observational lensing datasets**
* Explore **regression for mass distribution estimation**
* Investigate **transformer-based physics-informed architectures**
* Improve robustness to noise and domain shifts

---

## 🔷 Author

**Shrisha G K**
GSoC Applicant – ML4Sci DeepLense Project
