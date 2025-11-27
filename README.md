# Military Aircraft Recognition Using CNN 

This project is an end-to-end **Military Aircraft Recognition System** built using **Convolutional Neural Networks (CNN)**. The model identifies and classifies different military aircraft by processing annotated images, extracting object regions, training a deep learning model, and generating multiple visual evaluations to understand model performance.

---

## ⭐ Project Overview

- ⭐ This project uses **Convolutional Neural Networks (CNNs)** to classify different types of military aircraft from images.  
- ⭐ The system reads XML annotation files, extracts aircraft objects from images, preprocesses them, and trains a deep learning model.  
- ⭐ Hyperparameter tuning using **Keras Tuner (Random Search)** helps improve the model architecture automatically.  
- ⭐ Multiple visual analytics such as loss curves, accuracy curves, histograms, and pie charts are generated to give deeper insights into model training.  
- ⭐ The final system provides predictions, true vs predicted comparisons, and statistical improvement analysis.

---

## ⭐ Dataset Used

This project uses a publicly available military aircraft dataset with JPEG images and XML annotations.

### 🔗 Dataset Link  
👉 **Military Aircraft Detection Dataset (Kaggle)**  
https://www.kaggle.com/datasets/khlaifiabilel/military-aircraft-recognition-dataset/data

**What it contains:**

- ⭐ Military aircraft images  
- ⭐ Pascal-VOC style XML annotation files  
- ⭐ Bounding boxes for object locations  
- ⭐ Multiple types of aircraft categories  
- ⭐ Suitable for both **object detection and classification**

---

## ⭐ Techniques Used

This project uses a wide range of image processing and machine learning techniques:

### ⭐ Image Preprocessing Techniques
- ⭐ Reading and parsing XML annotation files  
- ⭐ Extracting object bounding boxes from images  
- ⭐ Padding images to maintain aspect ratio  
- ⭐ Resizing images to uniform dimensions  
- ⭐ Converting images to NumPy arrays  
- ⭐ Normalizing pixel values  
- ⭐ Label encoding using **OneHotEncoder**

### ⭐ Dataset Handling Techniques
- ⭐ file-based train/test reading  
- ⭐ Filtering missing or mismatched annotations  
- ⭐ Preparing X_train, X_test, y_train, y_test  
- ⭐ Train-test splitting using **Sklearn**

### ⭐ Hyperparameter Tuning Techniques
- ⭐ Using **Keras Tuner – RandomSearch** to find:
  - Best number of convolution filters  
  - Best kernel size  
  - Best dense layer size  
  - Best dropout rate  
  - Learning rate tuning  

### ⭐ Visualization Techniques
- ⭐ Line graphs (loss and accuracy)  
- ⭐ Histograms (loss distribution, accuracy distribution)  
- ⭐ Bar charts (epoch-wise comparison)  
- ⭐ Pie charts (improvement vs non-improvement)  
- ⭐ Prediction comparison grid (True vs Predicted)

---

## ⭐ Type of Model Used

The project uses a **Convolutional Neural Network (CNN)** built using TensorFlow/Keras.

### ⭐ Model Architecture (Summary)

- ⭐ **Conv2D Layers** — extract spatial features from images  
- ⭐ **AveragePooling Layers** — reduce dimensionality while keeping key information  
- ⭐ **Flatten Layer** — convert feature maps into a vector  
- ⭐ **Dense Layers** — learn high-level patterns  
- ⭐ **Dropout Layer** — prevent overfitting  
- ⭐ **Softmax Output Layer** — classify into multiple aircraft categories  

### ⭐ Model Optimization
- ⭐ Optimizer: **Adam**  
- ⭐ Loss Function: **Categorical Crossentropy**  
- ⭐ Metrics: **Accuracy**  
- ⭐ Hyperparameters tuned using **Keras Tuner**

---

## ⭐ Key Features of the Project

- ⭐ Automatic reading of image & annotation files  
- ⭐ Aircraft extraction using bounding boxes  
- ⭐ Advanced preprocessing pipeline  
- ⭐ CNN with hyperparameter tuning  
- ⭐ Clear visualization of training results  
- ⭐ Prediction with true vs predicted values  
- ⭐ Multiple graph types for metrics analysis  
- ⭐ Histogram, Bar chart, and Pie chart explanations  
- ⭐ Fully explainable and reproducible deep-learning workflow  

---

## ⭐ How the System Works (Step-By-Step)

1. ⭐ Load images and XML annotation files  
2. ⭐ Extract the aircraft region using bounding box coordinates  
3. ⭐ Preprocess extracted images (padding, resizing, normalization)  
4. ⭐ Encode labels using One-Hot Encoding  
5. ⭐ Split data into training and testing sets  
6. ⭐ Build the CNN model  
7. ⭐ Tune parameters using Keras Tuner’s RandomSearch  
8. ⭐ Train the model on preprocessed images  
9. ⭐ Evaluate and visualize performance  
10. ⭐ Make predictions and display results  
11. ⭐ Plot histograms, bar charts, and pie charts for deeper insights  

---
