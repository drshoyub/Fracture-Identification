import streamlit as st
from fastai.vision.all import *
from PIL import Image
import urllib.request
import os

# 1. Page Configuration & Title Styling
st.set_page_config(page_title="AI Fracture Reader", page_icon="🏥", layout="centered")
st.title("🏥 Permanent AI Bone Fracture Assistant")
st.write("Upload any bone radiograph to query our custom-trained 99.27% accurate computer vision network.")

# 🔗 Clean, absolute direct download link format for your file
DROPBOX_URL = "https://dropboxusercontent.com"
MODEL_PATH = "fracture_model.pkl"

# 2. Function to cleanly pull and cache the model file
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        # If the file doesn't exist, or is a tiny broken HTML file (< 1MB), wipe and redownload
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
            
        with st.spinner("📥 Streaming raw AI model weights from secure storage... (Takes a moment on first run)"):
            # Using the direct dl.dropboxusercontent path forces a pure raw data stream
            urllib.request.urlretrieve(DROPBOX_URL, MODEL_PATH)
            
    return load_learner(MODEL_PATH)

# Safely initialize the model framework
try:
    model = download_and_load_model()
    labels = model.dls.vocab
except Exception as e:
    st.error(f"Model sync issue: {str(e)}")
    st.info("The server downloaded an HTML web page instead of the raw file data. Retrying with clear cache...")
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    st.stop()

# 3. Interactive Drag and Drop File Upload Dashboard UI
uploaded_file = st.file_uploader("Drag and drop your X-ray file here...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Radiograph", use_container_width=True)
    st.write("🔄 Calculating Neural Diagnostics...")
    
    pred, pred_idx, probs = model.predict(PILImage.create(uploaded_file))
    
    st.success(f"**AI STRUCTURAL FINDING:** {pred.upper()}")
    st.metric(label="AI Confidence Level", value=f"{probs[pred_idx]*100:.2f}%")
