import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="RiceVision AI | Grain Classification System",
    page_icon="🍚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS DESIGN SYSTEM
# =========================================================
st.markdown("""
<style>
    /* CSS Variables */
    :root {
        --bg-main: #1A1E17;
        --bg-card: #252B20;
        --bg-sidebar: #141712;
        --accent-primary: #A3D175;
        --accent-hover: #B5DC8E;
        --text-primary: #E2EFE0;
        --text-secondary: #9DB091;
        --border-color: rgba(163, 209, 117, 0.18);
        --border-highlight: rgba(163, 209, 117, 0.45);
    }

    /* Main Container Reset */
    .stApp {
        background-color: var(--bg-main);
        color: var(--text-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Adjusted padding-top to move the header slightly down */
    .block-container {
        max-width: 1200px;
        padding-top: 4.5rem;
        padding-bottom: 3rem;
    }

    /* Dynamic Headings */
    h1, h2, h3, h4 {
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: var(--bg-sidebar);
        border-right: 1px solid var(--border-color);
    }
    
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: var(--accent-primary) !important;
    }

    /* Layout Cards */
    .ui-card {
        background-color: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }

    .ui-card-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: var(--accent-primary);
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Header Component */
    .header-container {
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 2.5rem;
    }

    .main-title {
        color: var(--accent-primary);
        font-size: 2.75rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 6px;
    }

    .subtitle {
        color: var(--text-secondary);
        font-size: 1.05rem;
        font-weight: 400;
    }

    /* Prediction Display Card */
    .prediction-card {
        background: linear-gradient(145deg, #2A3224, #21271C);
        border: 1px solid var(--border-highlight);
        border-radius: 16px;
        padding: 32px 24px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .prediction-badge {
        display: inline-block;
        background: rgba(163, 209, 117, 0.12);
        color: var(--accent-primary);
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        padding: 6px 14px;
        border-radius: 20px;
        border: 1px solid rgba(163, 209, 117, 0.25);
        margin-bottom: 12px;
    }

    .prediction-result {
        color: #FFFFFF;
        font-size: 2.25rem;
        font-weight: 800;
        margin: 8px 0;
    }

    .confidence-score {
        color: var(--text-secondary);
        font-size: 0.95rem;
    }

    .confidence-score strong {
        color: var(--accent-primary);
        font-weight: 700;
    }

    /* Custom Probability Meters */
    .meter-container {
        margin-bottom: 14px;
    }

    .meter-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.88rem;
        margin-bottom: 5px;
        color: var(--text-primary);
    }

    .meter-bar-bg {
        background-color: rgba(255, 255, 255, 0.06);
        border-radius: 6px;
        height: 8px;
        width: 100%;
        overflow: hidden;
    }

    .meter-bar-fill {
        background: linear-gradient(90deg, #7FA658, var(--accent-primary));
        height: 100%;
        border-radius: 6px;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .meter-bar-fill.top-rank {
        background: linear-gradient(90deg, var(--accent-primary), #C8E8A3);
    }

    /* Streamlit Uploader Styling Override */
    [data-testid="stFileUploader"] {
        background-color: var(--bg-card);
        border: 1px dashed var(--border-highlight);
        border-radius: 12px;
        padding: 12px;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent-primary);
    }

    /* Footer */
    .footer {
        text-align: center;
        color: var(--text-secondary);
        font-size: 0.82rem;
        margin-top: 60px;
        padding-top: 24px;
        border-top: 1px solid var(--border-color);
        opacity: 0.8;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# MODEL SETUP & PREPROCESSING
# =========================================================
CLASS_NAMES = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]
IMAGE_SIZE = (150, 150)

@st.cache_resource
def load_classification_model():
    """Load and cache the TensorFlow CNN model."""
    return tf.keras.models.load_model("rice_cnn_model.keras")

try:
    model = load_classification_model()
except Exception:
    st.error("❌ **Model Loading Failed**: Unable to find or initialize `rice_cnn_model.keras`.")
    st.info("Please verify that the `.keras` model file is located in the working execution directory.")
    st.stop()

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess PIL Image into normalized array for inference."""
    img_resized = image.resize(IMAGE_SIZE)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

# =========================================================
# SIDEBAR NAVIGATION & INFO
# =========================================================
with st.sidebar:
    st.markdown("## 🍚 RiceVision AI")
    st.caption("Deep Learning Grain Analysis Engine")
    st.markdown("---")

    st.markdown("### System Specifications")
    st.markdown("""
    * **Architecture:** Convolutional Neural Network (CNN)
    * **Target Varieties:** 5 Classes
    * **Optimization:** Adam
    """)

    st.markdown("### Supported Varieties")
    for name in CLASS_NAMES:
        st.markdown(f"• **{name}**")

    st.markdown("---")
    st.caption("Production Pipeline v1.0 • TensorFlow & Streamlit")

# =========================================================
# MAIN INTERFACE
# =========================================================
st.markdown("""
    <div class="header-container">
        <div class="main-title">🍚 RiceVision AI</div>
        <div class="subtitle">Automated Rice Variety Identification & Quality Analytics</div>
    </div>
""", unsafe_allow_html=True)

# Overview Card
st.markdown("""
    <div class="ui-card">
        <div class="ui-card-header">🔍 Interactive Inference Dashboard</div>
        <p style="margin:0; color: var(--text-secondary); font-size: 0.95rem; line-height: 1.5;">
            Upload an image of a single  grains to receive real-time variety classification and probability distribution across supported categories: 
            <strong style="color: var(--text-primary);">Arborio, Basmati, Ipsala, Jasmine,</strong> and <strong style="color: var(--text-primary);">Karacadag</strong>.
        </p>
    </div>
""", unsafe_allow_html=True)

# Two-Column Workspace Layout
col_upload, col_results = st.columns(2, gap="large")

# ---------------------------------------------------------
# COLUMN 1: IMAGE INGESTION
# ---------------------------------------------------------
with col_upload:
    st.markdown("### 📤 Image Ingestion")
    
    uploaded_file = st.file_uploader(
        "Upload Grain Image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG",
        label_visibility="collapsed"
    )

    loaded_image = None
    if uploaded_file is not None:
        try:
            loaded_image = Image.open(uploaded_file).convert("RGB")
            st.image(
                loaded_image,
                caption="Uploaded Sample Source",
                use_container_width=True
            )
        except Exception:
            st.error("⚠️ Invalid image file. Please upload a standard image format.")
            loaded_image = None
   
# ---------------------------------------------------------
# COLUMN 2: INFERENCE & ANALYTICS
# ---------------------------------------------------------
with col_results:
    st.markdown("### 🤖 Model Inference")

    if loaded_image is None:
        st.markdown("""
            <div class="prediction-card" style="opacity: 0.6;">
                <div class="prediction-badge">System Idle</div>
                <div class="prediction-result" style="font-size: 1.8rem; color: var(--text-secondary);">
                    Awaiting Input
                </div>
                <div class="confidence-score">
                    Upload an image to display classification metrics.
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Automatic Prediction Pipeline
        with st.spinner("Processing tensor & running neural network..."):
            input_tensor = preprocess_image(loaded_image)
            raw_predictions = model.predict(input_tensor, verbose=0)[0]
            
            top_index = int(np.argmax(raw_predictions))
            top_class = CLASS_NAMES[top_index]
            top_confidence = float(raw_predictions[top_index]) * 100

        # Prediction Display
        st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-badge">Primary Classification</div>
                <div class="prediction-result">
                    🍚 {top_class}
                </div>
                <div class="confidence-score">
                    Model Confidence: <strong>{top_confidence:.2f}%</strong>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 Probability Distribution")

        # Custom Rendered Confidence Meters
        for idx, name in enumerate(CLASS_NAMES):
            prob_value = float(raw_predictions[idx])
            prob_percentage = prob_value * 100
            is_top = (idx == top_index)
            
            fill_class = "meter-bar-fill top-rank" if is_top else "meter-bar-fill"
            label_weight = "700" if is_top else "400"
            
            st.markdown(f"""
                <div class="meter-container">
                    <div class="meter-labels">
                        <span style="font-weight: {label_weight};">{name}</span>
                        <span style="font-weight: {label_weight}; color: var(--accent-primary);">{prob_percentage:.2f}%</span>
                    </div>
                    <div class="meter-bar-bg">
                        <div class="{fill_class}" style="width: {prob_percentage:.2f}%;"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
    <div class="footer">
        RiceVision AI • Industrial Computer Vision System • Built with TensorFlow & Streamlit
    </div>
""", unsafe_allow_html=True)