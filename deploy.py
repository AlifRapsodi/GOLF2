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

# Custom CSS to improve appearance
st.markdown("""
    <style>
    /* Background: Aurora Gradient */
    .main {
        padding: 2rem;
        background: linear-gradient(135deg, #6a11cb, #2575fc); /* Aurora Gradient */
        color: white; /* Set text color to white for contrast */
    }

    /* Buttons with Aurora Style */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #ff6a00, #ee0979);
        color: white;
        padding: 0.75rem;
        border-radius: 12px;
        border: none;
        font-size: 1rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #ee0979, #ff6a00);
        transform: scale(1.05); /* Slight zoom effect */
    }

    /* Prediction Box */
    .prediction-box {
        padding: 1.5rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #6a11cb, #2575fc); /* Aurora Gradient */
        color: white;
        margin: 1rem 0;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
    }

    /* Selectbox */
    .stSelectbox>div>div>div>div {
        background: linear-gradient(135deg, #ff5f6d, #ffc371); /* Blue Aurora Gradient */
        border-radius: 12px;
        padding: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.4);
        font-weight: bold;
    }

    /* File Uploader */
    .stFileUploader>div>div>div>div {
        background: linear-gradient(135deg, #ff5f6d, #ffc371); /* Warm Aurora Gradient */
        color: white;
        border-radius: 12px;
        padding: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.4);
        font-weight: bold;
    }

    /* Headings */
    .stMarkdown h3 {
        color: #ffd700; /* Vibrant gold */
    }

    /* Paragraphs */
    .stMarkdown p {
        color: #f8f9fa;
    }

    /* Images */
    .stImage>img {
        border-radius: 12px;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
        transition: transform 0.3s ease;
    }
    .stImage>img:hover {
        transform: scale(1.05); /* Zoom effect */
    }

    /* Plotly Chart */
    .stPlotlyChart {
        border-radius: 12px;
        background: linear-gradient(135deg, #36d1dc, #5b86e5); /* Blue Aurora Gradient */
        padding: 1rem;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)



# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Define labels
labels = ["Address", "Toe-up", "Mid-backswing", "Top", "Impact", "Mid-follow-through", "Finish"]

# Load ViT model based on user selection
model_paths = {
    "Base": r"src/base",
    "Tiny": r"src/tiny",
    "Small": r"src/small"
}

# Define transform for single image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def load_model(model_path):
    model = ViTForImageClassification.from_pretrained(model_path)
    model.eval()
    return model

def predict_image(model, image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Perform prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
        
    # Get prediction and confidence scores
    confidence_scores = probabilities.cpu().numpy()
    predicted_class = torch.argmax(probabilities).item()
    
    return predicted_class, confidence_scores

def plot_confidence_scores(confidence_scores):
    # Sort confidence scores and labels
    sorted_indices = np.argsort(confidence_scores)[::-1]
    sorted_scores = confidence_scores[sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]
    
    # Create horizontal bar chart using plotly
    fig = go.Figure(go.Bar(
        x=sorted_scores * 100,  # Convert to percentage
        y=sorted_labels,
        orientation='h',
        marker_color='rgba(108, 92, 231, 0.6)',
        text=[f'{score:.1f}%' for score in sorted_scores * 100],
        textposition='auto',
    ))
    
    fig.update_layout(
        title='Confidence Scores by Class',
        xaxis_title='Confidence (%)',
        yaxis_title='Class',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def main():
    # Header section
    st.title("🏌️ Golf Swing Phase Classifier")
    st.markdown("""
    This application uses Vision Transformer (ViT) to classify different phases of a golf swing.
    Upload your golf swing image and get instant classification results!
    """)
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Model selection with custom styling
        st.markdown("### Model Selection")
        model_type = st.selectbox(
            "Choose your preferred model",
            ["Base", "Tiny", "Small"],
            help="Select the model architecture you want to use for classification"
        )
        
        # Load selected model
        model_path = model_paths[model_type]
        model = load_model(model_path)
        
        # File uploader with instructions
        st.markdown("### Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a golf swing image (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of a golf swing"
        )
    
    # Display and process image
    if uploaded_file is not None:
        with col1:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width =True)
            
            # Predict button
            if st.button("Classify Swing Phase"):
                with st.spinner("Analyzing swing phase..."):
                    # Get prediction and confidence scores
                    predicted_class, confidence_scores = predict_image(model, image)
                    
                    # Display results in col2
                    with col2:
                        st.markdown("### Classification Results")
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h3 style='color: #6c5ce7;'>Predicted Phase: {labels[predicted_class]}</h3>
                            <p>Confidence: {confidence_scores[predicted_class]*100:.1f}%</p>
                            <p>Analyzed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display confidence scores plot
                        st.plotly_chart(plot_confidence_scores(confidence_scores), use_container_width=True)
                        
                        # Display additional information
                        st.markdown("### About the Classification")
                        st.markdown("""
                        The confidence scores show the model's certainty level for each possible
                        swing phase. Higher percentages indicate greater confidence in that
                        particular classification.
                        """)

if __name__ == "__main__":
    main()
