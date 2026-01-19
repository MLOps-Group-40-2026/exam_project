"""Streamlit frontend for Coffee Leaf Disease Classifier."""

import requests
import streamlit as st
from PIL import Image

API_URL = "https://coffee-api-485178670977.europe-west1.run.app"

st.set_page_config(
    page_title="Coffee Leaf Disease Classifier",
    layout="centered",
)

st.title("Coffee Leaf Disease Classifier")
st.markdown(
    """
    Upload an image of a coffee leaf to detect diseases.

    **Supported classes:** Healthy, Miner, Phoma, Red Spider Mite, Rust
    """
)

uploaded_file = st.file_uploader(
    "Choose a coffee leaf image...",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Classify", type="primary"):
        with st.spinner("Analyzing..."):
            try:
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
                response.raise_for_status()
                result = response.json()

                st.success(f"**Prediction:** {result['prediction']}")
                st.metric("Confidence", f"{result['confidence']:.1%}")

                st.subheader("All Probabilities")
                for disease, prob in sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True):
                    st.progress(prob, text=f"{disease}: {prob:.1%}")

            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to API: {e}")
            except Exception as e:
                st.error(f"Error: {e}")

st.divider()
st.caption(f"API: {API_URL} | [API Docs]({API_URL}/docs)")
