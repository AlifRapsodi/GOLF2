import streamlit as st
import torch
from torchvision import transforms
from transformers import ViTForImageClassification
from PIL import Image
import numpy as np
import random
import plotly.graph_objects as go
from datetime import datetime

# Set page config for better appearance
st.set_page_config(
    page_title="Golf Swing Classifier",
    page_icon="🏌️",
    layout="wide"
)

# Custom CSS with modern pink-purple theme
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        background-color: #faf5ff;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #e879f9 0%, #c084fc 100%);
        color: white;
        padding: 0.75rem;
        border-radius: 15px;
        border: none;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(192, 132, 252, 0.2);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(192, 132, 252, 0.3);
        background: linear-gradient(135deg, #d946ef 0%, #a855f7 100%);
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.9);
        margin: 1.5rem 0;
        box-shadow: 0 8px 16px rgba(192, 132, 252, 0.15);
        border: 1px solid rgba(192, 132, 252, 0.1);
        backdrop-filter: blur(12px);
    }
    .stSelectbox>div>div>div>div {
        background-color: white;
        border-radius: 15px;
        padding: 0.5rem;
        border: 1px solid rgba(192, 132, 252, 0.2);
        box-shadow: 0 4px 6px rgba(192, 132, 252, 0.1);
    }
    .stFileUploader>div>div>div>div {
        background-color: white;
        border-radius: 15px;
        padding: 1rem;
        border: 2px dashed rgba(192, 132, 252, 0.3);
        transition: all 0.3s ease;
    }
    .stFileUploader>div>div>div>div:hover {
        border-color: #c084fc;
        background-color: rgba(192, 132, 252, 0.05);
    }
    .stMarkdown h1 {
        color: #9333ea;
        font-weight: 800;
        margin-bottom: 1.5rem;
    }
    .stMarkdown h3 {
        color: #a855f7;
        font-weight: 600;
        margin: 1rem 0;
    }
    .stMarkdown p {
        color: #4b5563;
        line-height: 1.6;
    }
    .stImage>img {
        border-radius: 20px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(192, 132, 252, 0.1);
    }
    .stPlotlyChart {
        border-radius: 20px;
        background: white;
        padding: 1rem;
        box-shadow: 0 8px 16px rgba(192, 132, 252, 0.15);
        border: 1px solid rgba(192, 132, 252, 0.1);
    }
    div[data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
    }
    .stApp {
        background: linear-gradient(135deg, #fdf4ff 0%, #faf5ff 100%);
    }
    </style>
    """, unsafe_allow_html=True)

# Rest of the code remains the same until the plot_confidence_scores function

def plot_confidence_scores(confidence_scores):
    # Sort confidence scores and labels
    sorted_indices = np.argsort(confidence_scores)[::-1]
    sorted_scores = confidence_scores[sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]
    
    # Create horizontal bar chart with updated colors
    fig = go.Figure(go.Bar(
        x=sorted_scores * 100,
        y=sorted_labels,
        orientation='h',
        marker=dict(
            color='rgba(192, 132, 252, 0.6)',
            line=dict(color='rgba(168, 85, 247, 0.8)', width=1)
        ),
        text=[f'{score:.1f}%' for score in sorted_scores * 100],
        textposition='auto',
    ))
    
    fig.update_layout(
        title={
            'text': 'Confidence Scores by Class',
            'font': {'color': '#9333ea', 'size': 20}
        },
        xaxis_title='Confidence (%)',
        yaxis_title='Class',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#4b5563'),
        xaxis=dict(gridcolor='rgba(192, 132, 252, 0.1)'),
        yaxis=dict(gridcolor='rgba(192, 132, 252, 0.1)')
    )
    
    return fig

def main():
    # Header section with updated styling
    st.title("🏌️ Golf Swing Phase Classifier")
    st.markdown("""
    <div style='background: white; padding: 1.5rem; border-radius: 20px; box-shadow: 0 4px 6px rgba(192, 132, 252, 0.1); margin-bottom: 2rem;'>
    This application uses Vision Transformer (ViT) to classify different phases of a golf swing.
    Upload your golf swing image and get instant classification results!
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Model Selection")
        model_type = st.selectbox(
            "Choose your preferred model",
            ["Base", "Tiny", "Small"],
            help="Select the model architecture you want to use for classification"
        )
        
        model_path = model_paths[model_type]
        model = load_model(model_path)
        
        st.markdown("### Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a golf swing image (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of a golf swing"
        )
    
    if uploaded_file is not None:
        with col1:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("Classify Swing Phase"):
                with st.spinner("✨ Analyzing swing phase..."):
                    predicted_class, confidence_scores = predict_image(model, image)
                    
                    with col2:
                        st.markdown("### Classification Results")
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h3 style='color: #a855f7; margin-bottom: 1rem;'>Predicted Phase: {labels[predicted_class]}</h3>
                            <p style='font-size: 1.1rem; color: #6b7280;'>Confidence: <span style='color: #9333ea; font-weight: 600;'>{confidence_scores[predicted_class]*100:.1f}%</span></p>
                            <p style='font-size: 0.9rem; color: #6b7280;'>Analyzed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.plotly_chart(plot_confidence_scores(confidence_scores), use_container_width=True)
                        
                        st.markdown("### About the Classification")
                        st.markdown("""
                        <div style='background: white; padding: 1.5rem; border-radius: 20px; box-shadow: 0 4px 6px rgba(192, 132, 252, 0.1);'>
                        The confidence scores show the model's certainty level for each possible
                        swing phase. Higher percentages indicate greater confidence in that
                        particular classification.
                        </div>
                        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
