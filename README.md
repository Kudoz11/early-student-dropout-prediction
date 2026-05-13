# 🎓 Student Dropout Early Warning System (EWS)
### *End-to-End Machine Learning Project: From Data Analysis to Deployment*

---

## 🌟 Overview
Student dropout is a critical issue in higher education. This project provides a **Data-Driven solution** to identify at-risk students right after their first semester. 

By using **XGBoost** and **Explainable AI (SHAP)**, this system helps academic counselors understand *why* a student might drop out (e.g., financial issues or academic performance) so they can intervene early.

---

## 🚀 Key Features
- **High Performance:** Achieved ~89% accuracy using the XGBoost classifier.
- **Explainable AI:** Uses SHAP to provide transparency behind every prediction.
- **Live Dashboard:** Built an interactive UI with **Streamlit** for real-time risk assessment.
- **Data Pipeline:** Includes full Exploratory Data Analysis (EDA) and feature engineering.

---

## 🛠️ Tech Stack
- **Languages:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib, Seaborn
- **Deployment:** Streamlit
- **Notebook:** Jupyter Notebook for model training and validation

---

## 📂 Project Structure
- `student_dropout_analysis.ipynb`: Full training pipeline and EDA.
- `app.py`: Streamlit application code for the web interface.
- `xgboost_student_model.pkl`: The trained and serialized ML model.
- `dataset.csv`: Cleaned data used for training.

---

## 👨‍💻 How to Run Locally
1. Clone the repo: `git clone https://github.com/Kudoz11/early-student-dropout-prediction.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

---
*Developed by Kundan Kudoz *
