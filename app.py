import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Fraudulent Transaction Detection",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Fraudulent Transaction Detection")
st.write("Enter transaction details and use the Logistic Regression model to predict whether the transaction is fraudulent.")

st.info(
    "This app follows the preprocessing used in your notebook: "
    "LabelEncoder for MerchantCategory, CustomerGender and TransactionLocation, "
    "StandardScaler for features, and Logistic Regression for prediction."
)

# ---------------------------------------------------------
# Load training data
# ---------------------------------------------------------
st.sidebar.header("1. Load Training Data")
df = pd.read_csv("fraud_data.csv")

required_columns = [
    "TransactionID",
    "TransactionAmount",
    "TransactionTime",
    "MerchantCategory",
    "CustomerAge",
    "CustomerGender",
    "CustomerIncome",
    "TransactionLocation",
    "PreviousFraudCount",
    "Fraud"
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"Missing columns: {', '.join(missing_columns)}")
    st.stop()

# Remove rows with missing values, if any
df = df.dropna().copy()

# ---------------------------------------------------------
# Train model exactly in the style of the notebook
# ---------------------------------------------------------
label_encoders = {}

categorical_columns = [
    "MerchantCategory",
    "CustomerGender",
    "TransactionLocation"
]

for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    label_encoders[column] = le

X = df.drop(["TransactionID", "Fraud"], axis=1)
y = df["Fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test_scaled))

st.sidebar.success(f"Model trained successfully\nAccuracy: {accuracy:.2%}")

# ---------------------------------------------------------
# Prediction form
# ---------------------------------------------------------
st.header("2. Enter Transaction Details")

col1, col2 = st.columns(2)

with col1:
    transaction_amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=2500.0,
        step=10.0
    )

    transaction_time = st.number_input(
        "Transaction Time (hour)",
        min_value=0.0,
        max_value=23.99,
        value=12.0,
        step=0.01
    )

    customer_age = st.number_input(
        "Customer Age",
        min_value=18,
        max_value=100,
        value=40
    )

    customer_income = st.number_input(
        "Customer Income",
        min_value=0.0,
        value=100000.0,
        step=1000.0
    )

with col2:
    merchant_options = list(label_encoders["MerchantCategory"].classes_)
    gender_options = list(label_encoders["CustomerGender"].classes_)
    location_options = list(label_encoders["TransactionLocation"].classes_)

    merchant_category = st.selectbox(
        "Merchant Category",
        merchant_options
    )

    customer_gender = st.selectbox(
        "Customer Gender",
        gender_options
    )

    transaction_location = st.selectbox(
        "Transaction Location",
        location_options
    )

    previous_fraud_count = st.number_input(
        "Previous Fraud Count",
        min_value=0,
        value=0,
        step=1
    )

predict_button = st.button(
    "🔍 Check Transaction",
    use_container_width=True
)

if predict_button:
    # Encode categorical inputs using the same encoders fitted during training
    try:
        merchant_encoded = label_encoders["MerchantCategory"].transform(
            [merchant_category]
        )[0]

        gender_encoded = label_encoders["CustomerGender"].transform(
            [customer_gender]
        )[0]

        location_encoded = label_encoders["TransactionLocation"].transform(
            [transaction_location]
        )[0]
    except ValueError as e:
        st.error(f"Unknown category: {e}")
        st.stop()

    # IMPORTANT:
    # The order must match X.columns from the notebook.
    input_data = pd.DataFrame([{
        "TransactionAmount": transaction_amount,
        "TransactionTime": transaction_time,
        "MerchantCategory": merchant_encoded,
        "CustomerAge": customer_age,
        "CustomerGender": gender_encoded,
        "CustomerIncome": customer_income,
        "TransactionLocation": location_encoded,
        "PreviousFraudCount": previous_fraud_count
    }])

    input_data = input_data[X.columns]

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.divider()

    if prediction == 1:
        st.error("🚨 FRAUDULENT TRANSACTION")
        st.metric("Fraud Probability", f"{probability:.2%}")
    else:
        st.success("✅ LEGITIMATE TRANSACTION")
        st.metric("Fraud Probability", f"{probability:.2%}")

    st.caption(
        "Prediction is generated by the Logistic Regression model trained on the uploaded dataset."
    )

st.divider()
st.caption("Fraud Detection App • Logistic Regression • Streamlit")
