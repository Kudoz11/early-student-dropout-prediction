import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Student Retention AI", page_icon="🎓", layout="wide")

# Custom CSS for a clean look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# 1. Model Load
@st.cache_resource
def load_model():
    return joblib.load('xgboost_student_model.pkl')

model = None
try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file 'xgboost_student_model.pkl' not found! Please make sure it is in the same folder as this app.")
except Exception as e:
    st.error(f"Error loading model: {e}")

# 2. CONSTANTS
FEATURE_NAMES = [
    'Marital status', 'Application mode', 'Application order', 'Course', 
    'Daytime/evening attendance', 'Previous qualification', 'Nacionality', 
    "Mother's qualification", "Father's qualification", "Mother's occupation", 
    "Father's occupation", 'Displaced', 'Educational special needs', 'Debtor', 
    'Tuition fees up to date', 'Gender', 'Scholarship holder', 'Age at enrollment', 
    'International', 'Curricular units 1st sem (credited)', 
    'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (evaluations)', 
    'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)', 
    'Curricular units 1st sem (without evaluations)', 'Unemployment rate', 
    'Inflation rate', 'GDP'
]

MEDIAN_VALUES = [1.0, 8.0, 1.0, 11.0, 1.0, 1.0, 1.0, 13.0, 14.0, 6.0, 8.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 20.0, 0.0, 0.0, 6.0, 8.0, 5.0, 12.4, 0.0, 11.1, 1.4, 0.32]

# --- SIDEBAR INPUTS ---
st.sidebar.header("📝 Student Profile")
with st.sidebar:
    st.subheader("Academic Performance")
    grade = st.slider("1st Sem Grade (0-20)", 0.0, 20.0, 12.4)
    approved = st.slider("Units Approved (1st Sem)", 0, 10, 5)
    
    st.subheader("Financial & Personal")
    debtor = st.selectbox("Is a Debtor?", ["No", "Yes"])
    tuition = st.selectbox("Tuition Up-to-Date?", ["Yes", "No"])
    scholarship = st.selectbox("Scholarship Holder?", ["No", "Yes"])
    age = st.number_input("Age at Enrollment", 17, 60, 20)
    gender = st.radio("Gender", ["Female", "Male"], horizontal=True)

# --- MAIN AREA ---
st.title("🎓 Student Persistence Prediction System")
st.info("This system uses a tuned XGBoost model to predict student dropout risk based on early-semester indicators.")

# Disable button functionality if model failed to load
analyze_button = st.button("🚀 Analyze Student Risk")

if analyze_button and model is None:
    st.warning("Model is not loaded, so predictions cannot be generated. Please check the model file on the server.")
elif analyze_button and model is not None:
    # Create baseline
    input_vector = np.array(MEDIAN_VALUES, dtype=float).copy()
    
    # Map inputs to correct indices
    input_vector[13] = 1.0 if debtor == "Yes" else 0.0
    input_vector[14] = 1.0 if tuition == "Yes" else 0.0
    input_vector[15] = 1.0 if gender == "Male" else 0.0
    input_vector[16] = 1.0 if scholarship == "Yes" else 0.0
    input_vector[17] = float(age)
    input_vector[22] = float(approved)
    input_vector[23] = float(grade)
    
    # Prediction
    try:
        test_df = pd.DataFrame([input_vector], columns=FEATURE_NAMES)
        prob = model.predict_proba(test_df)[0]
        dropout_prob = float(prob[1])
        grad_prob = float(prob[0])
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        dropout_prob = None
        grad_prob = None

    if dropout_prob is not None:
        # --- DISPLAY RESULTS ---
        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            # Balanced Threshold: 0.65 for Dropout
            if dropout_prob > 0.65:
                st.error("### ⚠️ Result: High Attrition Risk")
                st.metric("Prediction", "DROPOUT", f"{dropout_prob*100:.1f}% Risk", delta_color="inverse")
            else:
                st.success("### ✅ Result: Likely to Graduate")
                st.metric("Prediction", "GRADUATE", f"{grad_prob*100:.1f}% Confidence")

        with res_col2:
            st.write("**Risk Analysis Meter**")
            st.progress(min(max(dropout_prob, 0.0), 1.0))
            st.caption(f"Probability of Dropout: {dropout_prob*100:.2f}% | Threshold: 65%")

        # --- RESEARCH INSIGHTS ---
        st.markdown("### 🔍 Strategic Recommendations")
        rec_col1, rec_col2 = st.columns(2)
        
        with rec_col1:
            st.write("**Academic Status**")
            if approved < 4:
                st.warning("Low credit accumulation detected. Recommend supplementary classes.")
            elif grade > 14:
                st.success("Strong academic standing maintained.")
            else:
                st.write("Average academic performance.")

        with rec_col2:
            st.write("**Financial Status**")
            if debtor == "Yes" or tuition == "No":
                st.error("Financial liabilities detected. Suggest meeting with financial aid office.")
            elif scholarship == "No" and grade > 15:
                st.info("Eligible for academic scholarship consideration.")
            else:
                st.write("Financial status appears stable.")

# --- FOOTER ---
st.markdown("---")
st.caption("Developed by **Kundan Kumar** | B.Tech CSE | CGC Jhanjeri | Powered by XGBoost and Streamlit")
