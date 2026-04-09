# ♻️ RecycleVision Pro

### Garbage Image Classification using Deep Learning

---

## 🚀 Overview

RecycleVision Pro is an **AI-powered waste classification system** that uses Deep Learning and Computer Vision to identify garbage types such as plastic, metal, glass, paper, cardboard, and trash.

The application is built with a **production-style Streamlit dashboard**, featuring authentication, real-time predictions, and advanced analytics.

---

## 🎯 Problem Statement

Manual waste segregation is inefficient, error-prone, and time-consuming.
This project solves the problem by using a **CNN-based image classification model** to automate waste sorting.

---

## 💡 Key Features

### 🔐 Authentication

* Secure login system using session-based authentication

### 🏠 Project Overview Dashboard

* Model details
* Dataset insights
* Key metrics (accuracy, dataset size, classes)

### 📊 Advanced Analytics Dashboard

* Real dataset-based insights (no dummy data)
* Interactive filters:

  * Category selection
  * Minimum image count
* Multiple visualizations:

  * Pie chart
  * Bar chart
  * Line chart
  * Scatter plot
  * Area chart
  * Funnel chart

### 📷 Prediction System

* Upload garbage image
* Real-time prediction
* Confidence score
* Top-3 predictions visualization

### 📜 Prediction History

* Stores user predictions
* Download results as CSV

---

## 🧠 Tech Stack

| Category         | Tools Used                      |
| ---------------- | ------------------------------- |
| Language         | Python                          |
| Deep Learning    | TensorFlow, Keras               |
| Model            | MobileNetV2 (Transfer Learning) |
| Frontend         | Streamlit                       |
| Visualization    | Plotly                          |
| Image Processing | PIL, NumPy                      |

---

## 📊 Dataset

* Garbage Classification Dataset (Kaggle)
* Classes:

  * Cardboard
  * Glass
  * Metal
  * Paper
  * Plastic
  * Trash

---

## ⚙️ Workflow

1. Data Collection
2. Data Preprocessing

   * Image resizing (224x224)
   * Normalization
   * Augmentation
3. Model Training using Transfer Learning
4. Model Evaluation
5. Deployment using Streamlit

---

## 📈 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## 📁 Project Structure

```
RecycleVision/
│
├── app.py
├── model.h5
├── requirements.txt
├── README.md
│
├── dataset/
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
│
├── notebooks/
│   └── train_model.ipynb
│
├── utils/
│   ├── preprocessing.py
│   ├── prediction.py
│
├── assets/
│   └── screenshots/
│
└── .gitignore
```

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Deployment

You can deploy this project on Streamlit Cloud:

👉 https://streamlit.io/cloud

Steps:

1. Push project to GitHub
2. Connect GitHub to Streamlit Cloud
3. Select repository
4. Deploy `app.py`

---

## 📸 Screenshots

*Add your app screenshots here (Dashboard, Prediction, Analytics)*

---

## 🌍 Business Use Cases

* Smart recycling bins
* Municipal waste management systems
* Environmental analytics platforms
* Educational tools for waste segregation

---

## 🔮 Future Enhancements

* Grad-CAM (model explainability)
* REST API using FastAPI
* Role-based authentication
* Cloud deployment (AWS/GCP)
* Real-time camera integration

---

## 👨‍💻 Author

**Ashish**
Aspiring Data Analyst | AI Enthusiast

---

## ⭐ Support

If you found this project helpful, please ⭐ the repository!
