import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load Keras model
model = load_model("my_model.h5")

# Custom CSS for better UI
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .title {
            font-size:36px;
            font-weight:bold;
            color:#2c3e50;
            text-align:center;
        }
        .subtitle {
            font-size:18px;
            text-align:center;
            color:#7f8c8d;
        }
        .prediction-box {
            padding:20px;
            border-radius:15px;
            background-color:#ffffff;
            box-shadow:2px 2px 15px rgba(0,0,0,0.1);
            margin-top:20px;
            text-align:center;
        }
        .result {
            font-size:24px;
            font-weight:bold;
            color:#2980b9;
        }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown('<p class="title">🌸 Iris Species Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter flower measurements and get prediction instantly</p>', unsafe_allow_html=True)

# Input form
st.markdown("### 🔹 Enter Flower Features")
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, step=0.1)
    petal_length = st.number_input('Petal Length (cm)', min_value=0.0, step=0.1)

with col2:
    sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, step=0.1)
    petal_width = st.number_input('Petal Width (cm)', min_value=0.0, step=0.1)

# Predict button
if st.button("🌼 Predict Species"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction, axis=1)[0]

    species = ["Iris Setosa", "Iris Versicolor", "Iris Virginica"]

    st.markdown(f"""
        <div class="prediction-box">
            <p class="result">✅ Predicted Species: {species[predicted_class]}</p>
        </div>
    """, unsafe_allow_html=True)

