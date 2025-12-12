# Computer-Aided Inspection System to Predict Food Quality Characteristics  
*A Machine Learning & Image Processing-Based Quality Evaluation System for Food Technology*
## Overview
This project implements a **Computer-Aided Inspection (CAI) system** designed to predict **physicochemical, textural, and sensory quality characteristics** of meat products using **non-destructive image analysis**.  
Images of meat samples (fresh, thawed, cooked, cured) are processed using **advanced texture feature extraction** and analyzed through **multiple machine-learning regression models**.  
By combining **GLCM, Gabor filters, Wavelet transforms**, and a **stacking ensemble**, the system delivers improved prediction accuracy across multiple quality attributes.
## Objectives
- Develop a **non-destructive**, fast, and cost-effective quality assessment tool.  
- Predict multiple **quality indicators** such as texture, color, tenderness, pH, moisture, and sensory scores.  
- Implement a **stacking ensemble** that outperforms individual models.  
- Provide an **automated AI pipeline** to support food quality inspection labs and industrial processing.
##  Key Components
### 🔹 **1. Image Feature Extraction**
- GLCM (Contrast, Correlation, Energy, Homogeneity)  
- Gabor features (frequency–orientation texture patterns)  
- Wavelet features (multi-resolution texture representation)  
### 🔹 **2. Machine Learning Models**
- Random Forest Regressor  
- Support Vector Regressor (SVR)  
- Gradient Boosting  
- **Stacking Ensemble** (final predictor)
### 🔹 **3. Prediction Output**
Predicts **multiple quality parameters** for each image sample.
## Project Structure
├── data/
│ ├── images/
│ └── labels.csv
├── src/
│ ├── features.py
│ ├── train.py
│ ├── predict.py
│ └── api.py
├── results/
├── models/
├── requirements.txt
**Technologies Used**
Python
scikit-learn
scikit-image
OpenCV
PyWavelets
Flask
