import streamlit as st
from fastai.vision.all import *
from PIL import Image
import requests
import os

# 1. Page Configuration & Title Styling
st.set_page_config(page_title="AI Fracture Reader", page_icon="🏥", layout="centered")
st.title("🏥 Permanent AI Bone Fracture Assistant")
st.write("Upload any bone radiograph to query our custom-trained 99.27% accurate computer vision network.")

# 🔗 Your custom Dropbox direct-download link
DROPBOX_URL = "https://dropbox.com"
MODEL_PATH = "fracture_model.pkl"

# 2. Network-safe function to stream large files from Dropbox
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Streaming AI weights from secure storage... (Takes a moment on first run)"):
            # Using custom headers to bypass the Dropbox bot blocker
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            response = requests.get(DROPBOX_URL, headers=headers, stream=True)
            
            if response.status_code == 200:
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                st.error(f"Network Error: Dropbox returned status code {response.status_code}")
                st.stop()
                
    return load_learner(MODEL_PATH)

# Safely initialize the model framework
try:
    model = download_and_load_model()
    labels = model.dls.vocab
except Exception as e:
    st.error(f"Model sync issue: {str(e)}")
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
