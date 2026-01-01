# A Smart and Multimodal Artificial Intelligence System for Intelligent Medical Assistance and Early Disease Detection
> Developed as part of an academic project at **Esprit School of Engineering**.

---

## 🩺 Overview

This project is an academic initiative conducted at Esprit School of Engineering, aiming to enhance healthcare accessibility, medical assistance, and early disease detection through the use of Artificial Intelligence and multimodal data analysis.

The proposed system integrates several intelligent components, including medical chatbots, medical document analysis, speech and image processing, and decision-support mechanisms. It is designed to assist both patients and healthcare professionals by providing understandable medical information, supporting symptom-based reasoning, and enabling early detection of critical health conditions.

By combining Natural Language Processing, Computer Vision, Speech Technologies, and AI-driven reasoning, the system contributes to a more inclusive, accessible, and proactive digital health ecosystem, particularly adapted to local linguistic and healthcare contexts.

---

🚀 Key Features

🧾 Intelligent Medical Document Analysis
Automatic interpretation of prescriptions, laboratory reports, and medical PDFs, including blood test analysis and explanation of abnormal values.

💬 AI-Based Medical Chatbots
Interactive chatbots capable of answering questions about prescribed medications, diseases, and medical procedures, as well as providing public health information.

🧠 Symptom-Oriented Medical Assistance
An intelligent chatbot that dynamically generates follow-up questions, identifies possible diseases, evaluates associated risks, and recommends the appropriate medical specialist.

🎤 Multilingual Voice Interaction
Support for speech recognition (ASR) and text-to-speech (TTS) in Arabic, French, English, and Tunisian Arabic (Latin writing), ensuring natural and inclusive interaction.

🧑‍⚕️ Early Stroke (AVC) Detection
Multimodal analysis combining facial asymmetry detection and speech classification to estimate stroke risk levels with interpretable explanations.

🦷 Dental Disease Detection
Automated detection of dental caries from dental radiographic images using computer vision techniques.

🫁 Pulmonary Disease Classification
AI-based analysis of chest X-ray images to classify conditions such as COVID-19, tuberculosis, viral pneumonia, bacterial pneumonia, and normal cases.

🌍 Context-Aware and Ethical Design
The system emphasizes explainability, data interpretation transparency, and ethical use of AI while preserving the central role of healthcare professionals.

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
medOrient/
├── project/                           # Main website (Django project)
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
```


---

## ⚙️ Getting Started

### 🛠️ Installation Steps

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/medOrient.git
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

## Acknowledgments

## Esprit School of Engineering

This project was completed under the guidance of [Professor Sonia MESBEH
](mailto:sonia.mesbeh@esprit.tn) and [Professor Jihene Hlel
](mailto:jihene.hlel@esprit.tn)   at Esprit School of Engineering.

## Acknowledgments
We thank the faculty, clinical collaborators, and peers for their valuable input and support throughout the development of DEEP_NEPHRO.

