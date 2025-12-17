import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="Ship Detection", layout="centered")

st.title("🚢 Ship Detection using YOLOv8")
st.write("Upload a satellite image to detect ships.")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    results = model.predict(source=temp_path, conf=0.25)

    result_image = results[0].plot()
    st.image(result_image, caption="Detected Ships", use_column_width=True)

    os.remove(temp_path)
