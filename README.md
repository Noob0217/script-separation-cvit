# Indic Script Separation Models (Printed Document)

🚀 **Indic Script Separation Models** is a machine learning project that focuses on **word-level script separation** in Indic printed documents.  
This work was developed as part of an OCR pipeline for Indic languages at **IIIT Hyderabad**.

---

## 📖 Project Overview
- Built **13 bilingual** and **12 multilingual** models for script separation.  
- Models were trained and fine-tuned using **ResNet50 architectures**.  
- Complete ML pipeline: **data curation → preprocessing → training → validation → testing → deployment**.  
- Integrated into the **IIIT Hyderabad iLock site** for real-world OCR applications.  
- Currently, only the **test script** is shared here (training scripts & weights are omitted intentionally).  

> ⚠️ *Note*: Model weights and training scripts are not included in this repo since it is unclear whether they can be shared. Only the **test script** and README are provided.

---

## 🛠️ Tech Stack  

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/ResNet50-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/CSV-000000?style=for-the-badge&logo=csv&logoColor=white"/>
  <img src="https://img.shields.io/badge/API%20Integration-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white"/>
  <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white"/>
  <img src="https://img.shields.io/badge/VS%20Code-007ACC?style=for-the-badge&logo=visual-studio-code&logoColor=white"/>
</p>



## 📂 Repository Structure
```
Indic-Script-Separation/
│── test_english_urdu.py     # Test script for evaluating models
│── README.md                # Project documentation (this file)
```
---

## 📊 Visualizations
### Confusion Matrix Example
The testing script generates a **confusion matrix** to visualize classification performance.

![Confusion Matrix](results/confusion_matrix_example.png)

### Loss Curves (Training/Validation)
The training pipeline plots **loss curves** to track training and validation progress.

![Loss Curves](results/loss_curves_example.png)

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone <repo_url>
   cd Indic-Script-Separation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the test script:
   ```bash
   python test_english_urdu.py
   ```

---

## 🏆 Results
- Models achieved **high accuracy** on bilingual and multilingual datasets.  
- Scripts produce **CSV logs**, **confusion matrices**, and **loss plots** for better evaluation.

---

## 📌 Notes
- This repo is intended for showcasing the **workflow and testing** of Indic script separation models.  
- For full reproducibility, internal datasets and training weights are required (not shared here).

---

## 👨‍💻 Author
**Asim Ali**  
Machine Learning Engineer (Intern) at IIIT Hyderabad  

---
