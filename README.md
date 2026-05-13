# 🎓 Student Dropout Early Warning System (EWS)
### *Advanced Predictive Analytics & Explainable AI for Student Success*

---

## 📌 Executive Summary
Student attrition isn't just an academic failure; it's a systemic challenge. This project introduces a **Data-Driven Intervention System** designed to identify "at-risk" students immediately after their **first semester**. 

By shifting from reactive to proactive monitoring, educational institutions can implement support strategies while they still have the window of opportunity to influence a student's journey.

---

## 🚀 Why This Project Stands Out
- **Early-Stage Focus:** Unlike models that use final-year data, this system predicts outcomes using only **initial academic indicators**, making the insights truly actionable.
- **Handling Data Realities:** Implemented **SMOTE** (Synthetic Minority Over-sampling Technique) to ensure the model doesn't overlook the "Dropout" class, which is typically the minority in educational datasets.
- **Explainable AI (XAI):** Integrated **SHAP** values to provide a transparent "Reasoning Engine." The model doesn't just flag a student; it highlights the top factors (e.g., financial debt or specific course grades) contributing to the risk.

---

## 🛠️ Technical Implementation
- **Core Engine:** XGBoost Classifier (Optimized for tabular data performance).
- **Metric Focus:** Prioritized **Recall (0.88)** over simple accuracy to minimize "False Negatives"—ensuring no struggling student goes unnoticed.
- **Deployment:** A production-ready **Streamlit Dashboard** that allows counselors to input student data and get instant risk assessments.
- **Tech Stack:** Python, Scikit-Learn, XGBoost, SHAP, Pandas, Matplotlib, Streamlit.

---

## 📈 Performance Metrics
| Metric | Result |
| :--- | :--- |
| **Model Accuracy** | 89.12% |
| **Dropout Recall** | 88.00% |
| **F1-Score** | 0.86 |

---

## 📂 Project Architecture
- `student_dropout_analysis.ipynb` — Full EDA, Feature Engineering, and Model Training.
- `app.py` — Streamlit Web Application source code.
- `xgboost_student_model.pkl` — Pre-trained production model.
- `dataset.csv` — Cleaned and processed training data.

---

## 👨‍💻 Author
**Kundan Kumar** *B.Tech in Computer Science & Engineering* [Chandigarh Group of Colleges (CGC), Jhanjeri, Mohali](https://github.com/Kudoz11/early-student-dropout-prediction)

---
