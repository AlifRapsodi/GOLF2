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
    "Base": r"\src\\base",
    "Tiny": r"\src\\tiny",
    "Small": r"\src\\small"
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
        _, predicted = torch.max(logits, 1)

    return labels[predicted.item()]

# Streamlit app
def main():
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
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Predict button
        if st.button("Predict"):
            with st.spinner("Classifying..."):
                prediction = predict_image(model, image)
            st.success(f"Predicted class: {prediction}")

if __name__ == "__main__":
    main()
