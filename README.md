# 🧠 Brain Tumor Detection with Deep Learning (CNN)

This project leverages Convolutional Neural Networks (CNNs) to automatically identify brain tumors from MRI scans. By applying deep learning methods, the system classifies images into two categories: Tumor Present and No Tumor.


## 📁 Dataset

The model was trained and validated using the BR35H Brain Tumor Detection Dataset from Kaggle, which provides labeled MRI images of patients with and without tumors.



## 🔍 Project Summary

* **Model Type**: CNN-based binary image classifier
* **Input Data**: MRI scan images (.jpg / .png)
* **Predicted Output**: "Tumor Detected" or "No Tumor"
* **Performance**: Achieved strong accuracy after training and validation
* **Core Libraries**: TensorFlow, Keras, OpenCV, NumPy, Matplotlib, and others



## 🚀 Key Features

* Supports MRI image uploads for quick predictions
* Lightweight, easy-to-understand network structure
* Training performed on a balanced dataset for reliable classification



## 🏗️ Network Design

* Three convolutional layers with ReLU activations
* MaxPooling layers for downsampling
* Dense fully connected layers for classification
* Final Sigmoid/Softmax layer for binary output



