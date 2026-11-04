import streamlit as st
from fastai.vision.all import *
from PIL import Image
import urllib.request
import os

st.set_page_config(page_title="AI Fracture Reader", page_icon="🏥", layout="centered")
st.title("🏥 Permanent AI Bone Fracture Assistant")
st.write("Upload any bone radiograph to query our custom-trained 99.27% accurate computer vision network.")

GOOGLE_DRIVE_FILE_ID = "1IqE4zZ0eEbeo6yYpGzYF4fS-21l_vYf_"
MODEL_PATH = "fracture_model.pkl"

@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Syncing AI weights from secure cloud storage... (This takes a few seconds on first load)"):
            download_url = f"https://google.com{GOOGLE_DRIVE_FILE_ID}"
            urllib.request.urlretrieve(download_url, MODEL_PATH)
    return load_learner(MODEL_PATH)

try:
    model = download_and_load_model()
    labels = model.dls.vocab
except Exception as e:
    st.error("Cloud storage sync issue. Please ensure your Google Drive file permissions are set to 'Anyone with the link'.")
    st.stop()

uploaded_file = st.file_uploader("Drag and drop your X-ray file here...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Radiograph", use_container_width=True)
    st.write("🔄 Calculating Neural Diagnostics...")
    pred, pred_idx, probs = model.predict(PILImage.create(uploaded_file))
    st.success(f"**AI STRUCTURAL FINDING:** {pred.upper()}")
    st.metric(label="AI Confidence Level", value=f"{probs[pred_idx]*100:.2f}%")
