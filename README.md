# ♻️ Waste Garbage Management System (AI)

An AI-powered web application that classifies waste images into multiple categories using Deep Learning.

---

## 🚀 Features

* 📤 Upload Image Detection
* 📸 Camera-based Detection
* 🔥 Top-3 AI Predictions
* 📊 Probability Distribution Chart
* 📈 Analytics Dashboard
* 📂 History Tracking + CSV Export
* ☁️ Auto Model Loading via Google Drive

---

## 🧠 Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **AI Model:** TensorFlow / Keras
* **Visualization:** Plotly
* **Deployment:** Streamlit Cloud

---

## 📂 Project Structure

```
project/
│
├── app.py
├── class_names.json
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone Repository

```
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Run Application

```
streamlit run app.py
```

---

## 🔗 Model Handling

The trained model is hosted on Google Drive and automatically downloaded at runtime using `gdown`.

---

## 📊 Example Output

* Waste Category Prediction
* Confidence Score (%)
* Top-3 Predictions
* Probability Chart

---

## 👨‍💻 Author

Developed by **Ashish**

---

## 🌱 Future Enhancements

* Explainable AI (XAI)
* Mobile optimization
* Multi-user authentication
* API integration

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
