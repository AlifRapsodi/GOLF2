import streamlit as st
import torch
from torchvision import transforms
from transformers import ViTForImageClassification
from PIL import Image
import numpy as np
import random

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
    transforms.Resize((224, 224)),  # Assuming input size is 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Function to load model
def load_model(model_path):
    model = ViTForImageClassification.from_pretrained(model_path)
    model.eval()
    return model

# Function to predict a single image
def predict_image(model, image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Preprocess the image
    image = transform(image).unsqueeze(0).to(device)

    # Perform prediction
    with torch.no_grad():
        outputs = model(image)
        logits = outputs.logits  # Access the logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        _, predicted = torch.max(logits, 1)

    return labels[predicted.item()], probs.cpu().numpy()[0]

# Streamlit app
def main():
    st.set_page_config(page_title="ViT Image Classification App", page_icon=":camera:", layout="wide")

    # Custom CSS
    st.markdown("""
        <style>
            .stApp {
                background-color: #f0f2f6;
            }
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                padding: 10px 24px;
                border-radius: 8px;
                border: none;
            }
            .stButton>button:hover {
                background-color: #45a049;
            }
            .stFileUploader>div>div>div>div {
                background-color: #ffffff;
                border-radius: 8px;
                padding: 20px;
            }
            .stMarkdown h1 {
                color: #4CAF50;
            }
            .stMarkdown h2 {
                color: #4CAF50;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("ViT Image Classification App")
    st.write("Upload an image and the model will classify it into one of the predefined categories.")

    # Model selection
    model_type = st.selectbox("Select Model Type", ["Base", "Tiny", "Small"])
    model_path = model_paths[model_type]
    model = load_model(model_path)

    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Predict button
        if st.button("Predict"):
            with st.spinner("Classifying..."):
                prediction, probs = predict_image(model, image)
            st.success(f"Predicted class: **{prediction}**")

            # Display probabilities
            st.subheader("Prediction Probabilities")
            for label, prob in zip(labels, probs):
                st.write(f"{label}: {prob:.4f}")

            # Visualize probabilities
            st.subheader("Probability Distribution")
            st.bar_chart(dict(zip(labels, probs)))

if __name__ == "__main__":
    main()
