# 🌱 Project H.A.R.N. — Plant Nutrient Deficiency Detector Using CNN

### Hydro-farming with Autonomous Regulation of Nutrients (H.A.R.N.)
A **deep learning-based system** that detects **plant nutrient deficiencies** — specifically **Nitrogen (N)**, **Phosphorus (P)**, and **Potassium (K)** — from **leaf images** using **Convolutional Neural Networks (CNNs)**.  
Built as a web application with **Streamlit** and **TensorFlow**, this project aims to assist farmers and researchers in identifying plant health issues early, improving crop yield and management efficiency.

---

## 🌾 Overview

Modern hydroponic and traditional farming systems often rely on external parameters like temperature, humidity, and pH for nutrient regulation. However, plants' **leaf color and texture** often reveal early signs of nutrient imbalance.  

**H.A.R.N.** leverages image processing and deep learning to detect these deficiencies directly from leaf images — providing a faster and more accurate method than manual inspection or chemical testing.

---

## 🧠 Features

- 🌿 Detects **N**, **P**, and **K** nutrient deficiencies in rice and spinach leaves.  
- 🧩 Built using **Convolutional Neural Networks (CNN)**.  
- ⚡ Provides **real-time predictions** via a **Streamlit web app**.  
- ☁️ Deployed on [Streamlit Cloud](https://plant-nutrient-deficiency-detector-using-cnn.streamlit.app/).  
- 📸 Simple interface — users can upload a leaf image and instantly view the diagnosis.  
- 🧾 Provides detailed class probabilities and prediction confidence.

---

## 🧬 Reference Paper

> [Using Deep Convolutional Neural Networks for Image-Based Diagnosis of Nutrient Deficiencies in Rice](https://www.hindawi.com/journals/cin/2020/7307252/)  
> — *Zhe Xu, Xi Guo, Anfan Zhu, Xiaolin He, Xiaomin Zhao, Yi Han, and Roshan Subedi.*

---

## 📊 Dataset

A labeled dataset of rice and spinach leaf images with visible nutrient deficiencies.  
📂 Available on Google Drive:  
[Plant Nutrient Dataset (Google Drive)](https://drive.google.com/drive/folders/1kfX8iL_A2MK-XbGqOowDwzDv0PWAO7Y6?usp=sharing)

Each image corresponds to one of the following classes:
- **Healthy**
- **Nitrogen Deficient**
- **Phosphorus Deficient**
- **Potassium Deficient**

---

## 🧰 Technologies Used

| Category | Tools / Libraries |
|-----------|------------------|
| **Language** | Python |
| **Machine Learning** | TensorFlow, Keras |
| **Data Handling** | NumPy, Pandas |
| **Visualization** | Matplotlib, OpenCV |
| **Web Framework** | Streamlit |
| **Version Control** | Git, GitHub |
| **Environment** | Virtualenv |

Badges:
![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)

---

## ⚙️ Installation Guide

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Kesavaraja67/Plant-Nutrient-Deficiency-Detector-Using-CNN.git
cd Plant-Nutrient-Deficiency-Detector-Using-CNN
