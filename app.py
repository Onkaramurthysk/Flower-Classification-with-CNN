import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import json

# Load class labels (Flower Names)
with open("class_labels.json", "r") as f:
    class_labels = json.load(f)

# Load the trained model
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)  # Use same architecture
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 102)  # 102 classes for Flowers102 dataset
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()
    model.to(device)  # Move model to GPU if available
    return model, device

model, device = load_model()

# Define image transformations (ensure consistency with training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# UI Design
st.title("🌸 Flower Classification with CNN")
st.write("Upload an image to predict the flower type!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)  # Move image to GPU if available

    # Make prediction
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        top3_conf, top3_classes = torch.topk(probabilities, 3)  # Get Top-3 Predictions

    # Display results
    st.write("### 🏆 Top Predictions:")
    for i in range(3):
        class_id = top3_classes[0][i].item()
        confidence = top3_conf[0][i].item() * 100
        flower_name = class_labels.get(str(class_id), f"Class {class_id}")
        st.write(f"**{i+1}. {flower_name}** - {confidence:.2f}% Confidence")
