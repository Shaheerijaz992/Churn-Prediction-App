import streamlit as st
import pandas as pd
import joblib
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
from pathlib import Path
BASE = Path(__file__).parent
model_files = {
    "Logistic Regression": BASE / "lr_model.pkl",
    "Random Forest": BASE / "rf_model.pkl",
    "Gradient Boosting": BASE / "gb_model.pkl",
    "KNN": BASE / "knn_model.pkl",
}
@st.cache_resource
def load_model(path):
    return joblib.load(path)
st.title("Customer Churn Predictor")
st.write("Choose a model, enter customer details, and predict whether the customer will leave.")
choice = st.selectbox("Choose model", list(model_files.keys()))
model = load_model(model_files[choice])
yes_no = ["Yes", "No"]
internet_opts = ["Yes", "No", "No internet service"]
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", yes_no)
    dependents = st.selectbox("Dependents", yes_no)
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone = st.selectbox("Phone Service", yes_no)
    multiple = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
with col2:
    security = st.selectbox("Online Security", internet_opts)
    backup = st.selectbox("Online Backup", internet_opts)
    protection = st.selectbox("Device Protection", internet_opts)
    support = st.selectbox("Tech Support", internet_opts)
    tv = st.selectbox("Streaming TV", internet_opts)
    movies = st.selectbox("Streaming Movies", internet_opts)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", yes_no)
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])
monthly = st.number_input("Monthly Charges", 18.0, 120.0, 65.0)
total = tenure * monthly
if st.button("Predict"):
    row = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }])
    pred = model.predict(row)[0]
    prob = model.predict_proba(row)[0][list(model.classes_).index("Yes")]
    st.subheader(f"Result ({choice})")
    if pred == "Yes":
        st.error(f"This customer is likely to CHURN (probability {prob:.1%})")
    else:
        st.success(f"This customer is likely to STAY (churn probability {prob:.1%})")
    # Sab models ka comparison
    st.subheader("All models compared")
    rows = []
    for name, path in model_files.items():
        m = load_model(path)
        p = m.predict_proba(row)[0][list(m.classes_).index("Yes")]
        rows.append({"Model": name,
                     "Prediction": "Churn" if m.predict(row)[0] == "Yes" else "Stay",
                     "Churn probability": f"{p:.1%}"})
    st.table(pd.DataFrame(rows))