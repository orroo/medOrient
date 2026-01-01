# DEEP_NEPHRO: AI-Powered Dialysis Monitoring and Optimization System

> A smart and scalable system for enhancing dialysis care through intelligent monitoring, clinical decision support, and treatment optimization.  
> Developed as part of an academic project at **Esprit School of Engineering**.

---

## 🩺 Overview

**DEEP_NEPHRO** is a university project from **Esprit School of Engineering** that aims to revolutionize dialysis treatment using artificial intelligence and real-time data processing. The system is designed to assist healthcare professionals by monitoring patients during dialysis sessions, predicting optimal session duration and frequency, and providing early warnings of patient deterioration.

By combining medical knowledge, machine learning, and web technologies, DEEP_NEPHRO helps deliver personalized and efficient dialysis care while reducing resource waste.

---

## 🚀 Features

- 📊 **Real-Time Monitoring**: Continuously tracks patient vitals during dialysis.
- 🧠 **AI-Driven Predictions**: Estimates ideal session duration and frequency.
- 🔔 **Early Deterioration Detection**: Warns staff about potential risks.
- 💡 **Explainable AI**: Uses SHAP values to ensure transparent decision-making.
- 🌱 **Eco-Aware Optimization**: Minimizes water consumption.

---

## 🧰 Tech Stack

### Frontend
- **Django** 

### Backend
- **Django** 
- **WebSockets** – Real-time data communication

### Machine Learning
- **PyTorch** – Deep learning models
- **scikit-learn** – Data preprocessing and classical ML
- **SHAP** – Model explainability
- **Adams** – Hyperparameter optimization
- **tensorflow** 
- **cuda** 
- **tensorflow** 
---

## 📁 Directory Structure
```bash
DEEP_NEPHRO/
├── biosignals_fb_to_mat_file-main/   # Script to keep IoT data updated for our monitoring
├── flask app/                         # Flask API for predictions
├── project/                           # Main website (Django project)
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
```


---

## ⚙️ Getting Started

### 🛠️ Installation Steps

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/DEEP_NEPHRO.git
cd DEEP_NEPHRO
```

2. **Set up Python environment and install dependencies**:
```bash
Set up Python backend:
python -m venv venv
venv/Scripts/activate
pip install -r requirements.txt
```

3. **Run the Django backend with Daphne (ASGI server)**:
```bash
cd project
daphne project.asgi:application
```

5. **Run the Flask app (if included as a microservice)**:
```bash
cd ../"flask app"
python app.py
```
6. **Run The IoT Data Updating Script**:
```bash
cd ../biosignals_fb_to_mat_file-main
pip install -r requirements.txt  // dependencies first
python fb2mat_synth.py
```
## Acknowledgments

## Esprit School of Engineering

This project was completed under the guidance of [Professor Sonia MESBEH
](mailto:sonia.mesbeh@esprit.tn) and [Professor Jihene Hlel
](mailto:jihene.hlel@esprit.tn)   at Esprit School of Engineering.

## Acknowledgments
We thank the faculty, clinical collaborators, and peers for their valuable input and support throughout the development of DEEP_NEPHRO.

