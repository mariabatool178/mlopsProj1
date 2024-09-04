import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st

model = joblib.load("liveModelV1.pkl")

data = pd.read_csv("mobile_price_range_data.csv")
X = data.iloc[:,:-1]
y = data.iloc[:, -1]

# Train test split for accuracy calculation on any the testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Make predictions for X_test set
y_pred = model.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Page title
st.title("Model Accuracy and Real-Time Prediction")

# Display Accuracy
st.write(f"Model {accuracy}")

# Real time prediction based on user inputs
st.header("Real-Time Prediction")
input_data = []
for col in X_test.columns:
    input_value = st.number_input(f'Input for feature {col}', value=0.0)
    input_data.append(input_value)

# Convert input data to dataframe
input_df = pd.DataFrame([input_data], columns=X_test.columns)

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.write(f'Prediction: {prediction[0]}')
