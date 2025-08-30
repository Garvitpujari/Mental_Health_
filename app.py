import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------
# Load dataset
# -----------------------
url = "https://raw.githubusercontent.com/Garvitpujari/Mental_Health_/refs/heads/main/cleaned_mental_health_dataset%20(1).csv"
df = pd.read_csv(url)

# -----------------------
# Encode categorical features
# -----------------------
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# -----------------------
# Split Data
# -----------------------
X = df.drop(columns=["Depression"])
y = df["Depression"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------
# Train Model
# -----------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# -----------------------
# Streamlit UI
# -----------------------
st.title("🧠 Mental Health Predictor")
st.write("This app predicts whether a person is likely to have **Depression** based on their inputs.")

st.subheader("📌 Model Accuracy")
st.write(f"✅ Accuracy: **{acc:.2f}**")

st.subheader("📋 Input Your Data")

# Dynamic input form
user_input = {}
for col in X.columns:
    if col in label_encoders:  
        # categorical: show options
        options = list(label_encoders[col].classes_)
        choice = st.selectbox(f"{col}", options)
        user_input[col] = label_encoders[col].transform([choice])[0]
    else:
        # numeric
        val = st.number_input(f"{col}", value=0)
        user_input[col] = val

# Convert to dataframe
input_df = pd.DataFrame([user_input])

if st.button("🔮 Predict"):
    prediction = model.predict(input_df)[0]
    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.error("⚠️ The model predicts: **Depression = YES**")
    else:
        st.success("✅ The model predicts: **Depression = NO**")

st.markdown("---")
st.subheader("ℹ️ Encoding Information")
st.write("The following parameters were label encoded (converted to numbers):")
for col, le in label_encoders.items():
    mapping = {cls: int(le.transform([cls])[0]) for cls in le.classes_}
    st.write(f"**{col}**: {mapping}")
