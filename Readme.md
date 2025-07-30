


# 🧠  Deepfake Face Detection using LSTM

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![UI](https://img.shields.io/badge/UI-Tkinter-lightblue.svg)
![Libraries](https://img.shields.io/badge/Libraries-OpenCV%20%7C%20TensorFlow%20%7C%20Keras-orange.svg)

> A GUI-based desktop application that detects deepfakes using a hybrid **CNN + LSTM** model by analyzing spatial and temporal features from video/image frames.

---

## 🚀 Overview

Deepfakes are AI-generated media where faces are altered to mimic someone else — a serious threat in the age of misinformation. This project uses a **CNN for spatial feature extraction** and an **LSTM for temporal pattern recognition** to classify media as *Real* or *Deepfake*.

Key components:

* GUI built with **Tkinter**
* Face extraction using **OpenCV Haarcascade**
* Model training with **Keras CNN + LSTM**
* Evaluation metrics: Accuracy, Precision, Recall, F1
* Live classification for both **images and videos**

---

## ✨ Features

* 📂 Upload and process a labeled dataset (CSV + face images)
* 🧠 Train an LSTM model with TimeDistributed CNN blocks
* 📈 Visualize model performance via a bar chart (Matplotlib)
* 🎞️ Upload any video/image and detect if it's a deepfake
* 🎯 Supports both `.jpg/.png` images and `.mp4/.avi` videos
* 🗂️ Automatic dataset preprocessing with class balancing
* 🧪 Artificially boosted metrics for demo presentations

---

## 🛠️ Tech Stack

| Category        | Libraries / Tools                              |
| --------------- | ---------------------------------------------- |
| GUI             | `Tkinter`                                      |
| Computer Vision | `OpenCV`, Haarcascade XML                      |
| Deep Learning   | `TensorFlow`, `Keras`, `LSTM`, `Conv2D`        |
| Data Handling   | `Pandas`, `NumPy`                              |
| Visualization   | `Matplotlib`                                   |
| Others          | `pickle`, `ImageDataGenerator`, `scikit-learn` |

---

## 📂 Dataset Structure

* CSV File: Must contain `videoname` and `label` columns.
* Images: Stored in `Dataset/faces_224/` with filenames matching `videoname` in the CSV.
* Haarcascade file: `model/haarcascade_frontalface_default.xml` must be placed in `/model`.

---

## 🔧 Installation Instructions

### 🖥️ Prerequisites

* Python 3.10+
* Install dependencies:

  ```bash
  pip install numpy pandas opencv-python matplotlib scikit-learn tensorflow keras
  ```

---

### ▶️ Running the Application

1. Ensure your dataset is structured properly.
2. Run the Python file:

   ```bash
   python main.py
   ```
3. Use the GUI to:

   * Upload dataset
   * Train the model
   * Detect deepfakes in images/videos

---


## 🧠 Model Details

* **Architecture**: CNN layers wrapped in `TimeDistributed`, followed by `LSTM`, then `Dense`.
* **Augmentation**: Real-time with `ImageDataGenerator`.
* **Loss Function**: `categorical_crossentropy`
* **Optimizer**: `Adam`
* **Training**: 80/20 train-test split with optional checkpointing

---

## 🧪 Example Demo Flow

1. Load CSV + face images via **Upload Dataset**
2. Click **Train LSTM Model** to build and save the model
3. Use **Video Based Deepfake Detection** to test on new media

---

## ⚠️ Important Notes

* Ensure images are exactly **32x32** (automatically resized)
* Haarcascade must be downloaded and placed inside the `/model` folder
* If no faces are detected in images/videos, check resolution and face visibility
* Boosted metrics are intended for **presentation only**


