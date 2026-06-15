from Data import generate_data
from Data import generate_data
import streamlit as st
import requests

import os

API_URL = os.getenv(
    "PREDICT_API_URL",
    "http://localhost:5000"
)

st.set_page_config(
    page_title="Housing Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Housing Price Prediction")

st.markdown("### Input House Features")

overall_qual = st.slider(
    "Overall Quality",
    min_value=1,
    max_value=10,
    value=5
)

garage_cars = st.slider(
    "Garage Capacity",
    min_value=0,
    max_value=4,
    value=2
)

gr_liv_area = st.number_input(
    "Ground Living Area (sq ft)",
    min_value=500,
    max_value=5000,
    value=1500,
    step=50
)

if st.button("Predict House Price"):

    payload = {
        "OverallQual": overall_qual,
        "GarageCars": garage_cars,
        "GrLivArea": gr_liv_area
    }

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=10
        )

        if response.status_code == 200:

            result = response.json()

            st.success(
                f"Predicted Price: ${result['prediction']:,.2f}"
            )

            st.info(
                f"Model Version: {result['model_version']}"
            )

        else:
            st.error(response.text)

    except Exception as e:
        st.error(str(e))