# 🎥📈 IntelliStreamAI – Cross-Domain Business Intelligence Platform

> Unifying OTT Streaming Analytics & Financial Signal Forecasting using AI 🚀  
> Built for OTT platforms, content strategists, and financial analysts.

---

## 📌 Project Description

**IntelliStreamAI** is a cutting-edge, end-to-end AI-powered platform that bridges **OTT consumption behavior** with **financial market intelligence**.

It leverages **machine learning, NLP, time-series forecasting, and explainable AI** to:

- Analyze viewer churn & content engagement patterns  
- Recommend personalized content  
- Predict script success using NLP  
- Forecast stock trends of media companies based on streaming + sentiment data  
- Provide explainable model decisions and actionable dashboards  

---

## 🎯 Key Features

- 📊 **OTT Churn & Behavior Analysis**  
- 🎬 **Script Success Predictor (NLP on Scripts)**  
- 💡 **Intelligent Recommendation Engine (Content + Viewer Matrix)**  
- 📈 **Stock Market Forecaster (ARIMA, Prophet, LSTM)**  
- 📃 **Explainable AI (SHAP, LIME Integration)**  
- 📁 **Multi-format Data Upload (CSV, Scripts, etc.)**  
- 🧐 **Chatbot-Style AI Assistant for Insight Queries**  
- 📊 **Real-time BI Dashboards**  

---

## 🧱 Architecture Overview

Modular, scalable and API-driven architecture using **Streamlit (UI)**, **FastAPI (backend)**, and multiple ML/NLP pipelines.

---

## 🔧 Tech Stack

| Module                  | Technologies Used                             |
| ----------------------- | --------------------------------------------- |
| Frontend (Web UI)       | Streamlit / React                             |
| Backend API             | FastAPI, Firebase Auth                        |
| Data Upload & Parsing   | Pandas, PyMuPDF, Python-Magic                 |
| Churn Prediction Engine | XGBoost, LightGBM, SHAP                       |
| Recommender System      | Neural CF, Matrix Factorization               |
| Script NLP Engine       | BERT, GPT embeddings, TextBlob                |
| Stock Forecasting       | ARIMA, Prophet, LSTM                          |
| Explainability          | SHAP, LIME                                    |
| Dashboarding            | Plotly, Matplotlib, Streamlit                 |
| Storage & DB            | Firebase, MongoDB (optionally PostgreSQL)     |
| CI/CD                   | GitHub Actions, Docker                        |
| Deployment              | Streamlit Cloud / Vercel + Dockerized Backend |

---

## 📂 Project Folder Structure

IntelliStreamAI/
├── frontend/ # Web UI (Streamlit or React)/n
├── backend/ # FastAPI backend & APIs/n
├── models/ # ML/NLP models/n
├── data/ # Raw, processed & external data/n
├── notebooks/ # Jupyter notebooks for EDA, NLP, modeling/n
├── dashboard/ # Dashboard rendering logic/n
├── utils/ # Helper modules, logger, configs/n
├── docs/ # Documentation & diagrams/n
├── tests/ # Unit tests/n
├── Dockerfile/n
├── docker-compose.yml/n
└── README.md/n


---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/IntelliStreamAI.git
cd IntelliStreamAI
2. Set up virtual environment
 
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
3. Install dependencies
 
pip install -r backend/requirements.txt
4. Run the backend
 
cd backend
uvicorn main:app --reload
5. Run the frontend
 
cd frontend/web
streamlit run app.py
✅ How to Use
Upload CSV (for user logs), scripts (PDF), or text reviews

Choose the module (Churn Prediction, Recommender, etc.)

View analysis, explanations, and dashboard

Download reports or insights

🧪 Testing
 
pytest tests/
📌 Contributing
We welcome contributions to improve IntelliStreamAI! Follow these steps:

Fork the repo

Create a branch (git checkout -b feature/my-feature)

Commit your changes (git commit -m 'Add some feature')

Push to the branch (git push origin feature/my-feature)

Create a Pull Request

📜 License
This project is licensed under the MIT License – see the LICENSE file for details.

🙌 Credits
Developed by [Your Name]
Part of MSc AI Flagship Project | Bioinformatics + Data Science | 2025
Contributors: GPT-4, TensorFlow, HuggingFace, PyTorch, Prophet, Streamlit, and Open Source ❤️

💬 Contact
📧 palyashika641@gmail.com
📍 LinkedIn: palyashika641
 







