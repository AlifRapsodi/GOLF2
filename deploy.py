# File: golf_swing_analyzer.py

import streamlit as st
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import torch
from torchvision import transforms

# Page Configuration
st.set_page_config(
    page_title="Golf Swing Analyzer Pro",
    page_icon="🏌️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Super Cute and Beautiful Color Palette
PRIMARY_COLOR = "#6C5B7B"  # Soft purple
SECONDARY_COLOR = "#C06C84"  # Blush pink
ACCENT_COLOR = "#F8B195"  # Peach
BACKGROUND_COLOR = "#F5F5F5"  # Light gray
CARD_COLOR = "#FFFFFF"  # White
GLASS_EFFECT = "backdrop-filter: blur(10px); background: rgba(255, 255, 255, 0.7);"  # Glassmorphism
TEXT_COLOR = "#4A4A4A"  # Soft dark gray
SHADOW_COLOR = "rgba(0, 0, 0, 0.1)"  # Soft shadow
TIP_BACKGROUND = "#FCE4EC"  # Light pink for tips
BACKGROUND_COLOR = "linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)"

def apply_custom_css():
    st.markdown(f"""
        <style>
        body {{
            background: {BACKGROUND_COLOR};
            font-family: 'Nunito', sans-serif;
            color: {TEXT_COLOR};
        }}
        .main-title-container {{
            text-align: center;
            margin: 2rem 0;
            animation: fadeIn 2s ease-in-out;
        }}
        .main-title {{
            color: {PRIMARY_COLOR};
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px {SHADOW_COLOR};
        }}
        .subtitle {{
            color: {SECONDARY_COLOR};
            font-size: 1.4rem;
            font-weight: 400;
        }}
        .upload-container {{
            {GLASS_EFFECT}
            padding: 2rem;
            border: 2px dashed {SECONDARY_COLOR};
            border-radius: 20px;
            text-align: center;
            margin-bottom: 1.5rem;
            animation: slideIn 1s ease-in-out;
            box-shadow: 0 4px 12px {SHADOW_COLOR};
        }}
        .card {{
            {GLASS_EFFECT}
            padding: 2rem;
            border-radius: 20px;
            margin-bottom: 1.5rem;
            animation: fadeIn 1.5s ease-in-out;
            box-shadow: 0 4px 12px {SHADOW_COLOR};
        }}
        .result-header {{
            color: {PRIMARY_COLOR};
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }}
        .confidence-bar {{
            height: 10px;
            border-radius: 5px;
            background: {BACKGROUND_COLOR};
            box-shadow: inset 0 2px 4px {SHADOW_COLOR};
        }}
        .confidence-bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, {SECONDARY_COLOR}, {ACCENT_COLOR});
            border-radius: 5px;
            animation: grow 1s ease-in-out;
        }}
        .footer {{
            text-align: center;
            color: {TEXT_COLOR};
            margin-top: 2rem;
            animation: fadeIn 2s ease-in-out;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        @keyframes slideIn {{
            from {{ transform: translateY(-20px); opacity: 0; }}
            to {{ transform: translateY(0); opacity: 1; }}
        }}
        @keyframes grow {{
            from {{ width: 0%; }}
            to {{ width: 100%; }}
        }}
        .swing-tips {{
            background: {TIP_BACKGROUND};
            padding: 1.5rem;
            border-radius: 20px;
            margin-top: 1.5rem;
            animation: fadeIn 1.5s ease-in-out;
            box-shadow: 0 4px 12px {SHADOW_COLOR};
        }}
        .swing-tips h3 {{
            color: {PRIMARY_COLOR};
            font-size: 1.4rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }}
        .swing-tips ul {{
            list-style-type: disc;
            padding-left: 1.5rem;
        }}
        .swing-tips li {{
            margin-bottom: 0.75rem;
            color: {TEXT_COLOR};
        }}
        .sidebar {{
            background: linear-gradient(135deg, {PRIMARY_COLOR}, {SECONDARY_COLOR});
            color: white;
            padding: 2rem;
            border-radius: 20px;
            margin: 1rem;
            box-shadow: 0 4px 12px {SHADOW_COLOR};
        }}
        .sidebar h1 {{
            color: white;
            font-size: 2rem;
            font-weight: 800;
            margin-bottom: 1rem;
        }}
        .sidebar p {{
            color: rgba(255, 255, 255, 0.9);
        }}
        .sidebar hr {{
            border-top: 1px solid rgba(255, 255, 255, 0.2);
        }}
        .file-uploader {{
            background: {CARD_COLOR};
            padding: 1.5rem;
            border-radius: 20px;
            box-shadow: 0 4px 12px {SHADOW_COLOR};
        }}
        .emoji {{
            font-size: 1.2rem;
            margin-right: 0.5rem;
        }}
        .result-card {{
            {GLASS_EFFECT}
            padding: 2rem;
            border-radius: 20px;
            margin-bottom: 1.5rem;
            animation: slideIn 1s ease-in-out;
            box-shadow: 0 4px 12px {SHADOW_COLOR};
        }}
        .result-card h2 {{
            color: {PRIMARY_COLOR};
            font-size: 2rem;
            font-weight: 800;
            margin-bottom: 1rem;
        }}
        .result-card p {{
            color: {TEXT_COLOR};
            font-size: 1.2rem;
        }}
        .result-card .confidence {{
            font-size: 1.4rem;
            font-weight: 700;
            color: {SECONDARY_COLOR};
        }}
        .result-card .confidence-bar {{
            height: 15px;
            border-radius: 10px;
            background: {BACKGROUND_COLOR};
            box-shadow: inset 0 2px 4px {SHADOW_COLOR};
        }}
        .result-card .confidence-bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, {SECONDARY_COLOR}, {ACCENT_COLOR});
            border-radius: 10px;
            animation: grow 1s ease-in-out;
        }}
        </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# Sidebar Configuration
def setup_sidebar():
    with st.sidebar:
        st.markdown(f"""
            <div class="sidebar">
                <h1>⛳ Golf Pro AI</h1>
                <p>Powered by Vision Transformer</p>
                <hr>
                <p>How to use:</p>
                <ol>
                    <li><span class="emoji">📤</span>Upload a golf swing image</li>
                    <li><span class="emoji">🤖</span>Wait for AI analysis</li>
                    <li><span class="emoji">📊</span>Review detailed results</li>
                </ol>
                <hr>
                <p style="font-size: 0.9rem;">Version 2.0</p>
            </div>
        """, unsafe_allow_html=True)

setup_sidebar()

# Main Title
def display_main_title():
    st.markdown(f"""
        <div class="main-title-container">
            <h1 class="main-title">Golf Swing Analyzer Pro</h1>
            <p class="subtitle">Advanced AI-Powered Golf Swing Phase Detection</p>
        </div>
    """, unsafe_allow_html=True)

display_main_title()

model_select = st.selectbox("Select a model", ["Tiny", "Small", "Base"])
if model_select == "Tiny":
    model_path = r"src/tiny"
elif model_select == "Small":
    model_path = r"src/small"
elif model_select == "Base":
    model_path = r"src/base"

# Load Model and Labels
@st.cache_resource
def load_model(model_path):
    feature_extractor = transforms.Compose([
        transforms.Resize((224, 224)),  # Assuming input size is 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[ 0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
    ])
    labels = ["Address", "Toe-up", "Mid-backswing", "Top",
              "Impact", "Mid-follow-through", "Finish"]
    model = ViTForImageClassification.from_pretrained(model_path)
    return feature_extractor, model, labels

feature_extractor, model, labels = load_model(model_path)

# Image Upload and Analysis
def analyze_image(uploaded_file):
    # Buka gambar yang diunggah
    image = Image.open(uploaded_file).convert('RGB')
    
    # Preprocess gambar menggunakan feature extractor
    inputs = feature_extractor(image).unsqueeze(0)
    
    # Pindahkan input ke perangkat yang sesuai (CPU atau GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = inputs.to(device)
    model.to(device)
    
    # Lakukan prediksi
    with torch.no_grad():
        outputs = model(inputs)
        logits = outputs.logits  # Ambil logits dari output model
        probs = torch.nn.functional.softmax(logits, dim=-1)  # Hitung probabilitas
        confidence, predicted = torch.max(probs, 1)  # Ambil nilai confidence dan prediksi
    
    # Ambil label prediksi dan nilai confidence
    predicted_label = labels[predicted.item()]
    confidence_value = confidence.item()
    
    # Tampilkan hasil prediksi
    display_results(predicted_label, confidence_value, probs)
    
    # Tampilkan skor confidence untuk semua label
    display_confidence_scores(labels, probs)
    
    # Tampilkan tips berdasarkan fase swing yang terdeteksi

def display_results(predicted_label, confidence, probs):
    st.markdown(f"""
        <div class="result-card">
            <h2>🏌️ {predicted_label}</h2>
            <p class="confidence">Confidence: {confidence:.1%}</p>
            <div class="confidence-bar">
                <div class="confidence-bar-fill" style="width: {confidence * 100}%;"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def display_confidence_scores(labels, probs):
    st.markdown('<h3>Detailed Confidence Scores</h3>', unsafe_allow_html=True)
    
    # Gabungkan labels dan probs menjadi list of tuples
    label_prob_pairs = list(zip(labels, probs[0]))
    
    # Urutkan berdasarkan confidence value (dari tertinggi ke terendah)
    sorted_pairs = sorted(label_prob_pairs, key=lambda x: x[1].item(), reverse=True)
    
    # Tampilkan hasil yang sudah diurutkan
    for label, prob in sorted_pairs:
        confidence_value = prob.item()
        st.markdown(f"""
            <div style="margin-bottom: 0.5rem;">
                <div style="display: flex; justify-content: space-between;">
                    <span>{label}</span>
                    <span>{confidence_value:.1%}</span>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-bar-fill" style="width: {confidence_value * 100}%;"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# Footer
def display_footer():
    st.markdown(f"""
        <div class="footer">
            <p>Powered by Vision Transformer (ViT) Technology</p>
            <p style="font-size: 0.9rem;">© 2024 Golf Swing Analyzer Pro</p>
        </div>
    """, unsafe_allow_html=True)

# Main Execution
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
if uploaded_file:
    analyze_image(uploaded_file)

display_footer()
