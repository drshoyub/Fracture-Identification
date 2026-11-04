import streamlit as st
from fastai.vision.all import *
from PIL import Image
import gdown
import os

# 1. Page Configuration & Title Styling
st.set_page_config(page_title="AI Fracture Reader", page_icon="🏥", layout="centered")
st.title("🏥 Permanent AI Bone Fracture Assistant")
st.write("Upload any bone radiograph to query our custom-trained 99.27% accurate computer vision network.")

# 2. Hardcoded File ID extracted from your specific Google Drive link
GOOGLE_DRIVE_FILE_ID = "1IqE4zZ0eEbeo6yYpGzYF4fS-21l_vYf_"
MODEL_PATH = "fracture_model.pkl"

# 3. Bypassing Google virus checkpoints with gdown
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Securely syncing AI weights from cloud storage... (Takes a moment on first run)"):
            # Construct standard Google download link format for gdown
            download_url = f"https://google.com{GOOGLE_DRIVE_FILE_ID}"
            # This tool automatically clicks through the virus warning wall
            gdown.download(download_url, MODEL_PATH, quiet=True)
            
    return load_learner(MODEL_PATH)

# Safely initialize the model framework
try:
    model = download_and_load_model()
    labels = model.dls.vocab
except Exception as e:
    st.error("Cloud storage sync issue. Please ensure your Google Drive file permissions are set to 'Anyone with the link'.")
    st.stop()

# 4. Interactive Drag and Drop File Upload Dashboard UI
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
