import streamlit as st
from fastai.vision.all import *
from PIL import Image
import urllib.request
import os

# 1. Page Configuration & Title Styling
st.set_page_config(page_title="AI Fracture Reader", page_icon="🏥", layout="centered")
st.title("🏥 Permanent AI Bone Fracture Assistant")
st.write("Upload any bone radiograph to query our custom-trained 99.27% accurate computer vision network.")

# 🔗 Your custom Dropbox direct-download link (Hardcoded with dl=1)
DROPBOX_URL = "https://www.dropbox.com/scl/fi/9enslazfyl9qf5huh53ef/fracture_model.pkl?rlkey=k1ksj3vlrooxspdj5zzkmrajy&st=r0f5lucu&dl=1"
MODEL_PATH = "fracture_model.pkl"

# 2. Optimized Function to download model weights from Dropbox
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Securely syncing AI weights from secure storage... (Takes a moment on first run)"):
            # Dropbox links stream flawlessly into Python environments
            urllib.request.urlretrieve(DROPBOX_URL, MODEL_PATH)
    return load_learner(MODEL_PATH)

# Safely initialize the model framework
try:
    model = download_and_load_model()
    labels = model.dls.vocab
except Exception as e:
    st.error("Model sync issue. Please double-check your Dropbox link or connection parameters.")
    st.stop()

# 3. Interactive Drag and Drop File Upload Dashboard UI
uploaded_file = st.file_uploader("Drag and drop your X-ray file here...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and render the incoming image on the webpage screen
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Radiograph", use_container_width=True)
    
    st.write("🔄 Calculating Neural Diagnostics...")
    
    # Execute the underlying FastAI binary prediction loop
    pred, pred_idx, probs = model.predict(PILImage.create(uploaded_file))
    
    # Display the final classification outputs cleanly
    st.success(f"**AI STRUCTURAL FINDING:** {pred.upper()}")
    st.metric(label="AI Confidence Level", value=f"{probs[pred_idx]*100:.2f}%")
