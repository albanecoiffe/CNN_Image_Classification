import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(page_title="CNN Image Classification", layout="centered")
st.title("\U0001F4F7 CNN Image Classification with PyTorch")
st.markdown("""
### 👩‍💻 Project Description
This project demonstrates how to use a **Convolutional Neural Network (CNN)** in **PyTorch** for image classification.

**Classes**:
- Social Security
- Driving License
- Others

**Goal**: Build and test a model that classifies an input image into one of the above categories.

**Tech stack**: `PyTorch`, `OpenCV`, `Pandas`, `NumPy`, `Matplotlib`, `Torchvision`
""")

# -------------------------------
# CNN Model definition (from notebook)
# -------------------------------
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 12 * 12, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

# -------------------------------
# Load model
# -------------------------------
import requests

@st.cache_resource
def load_model():
    model_path = "model2.pt"
    url = "https://huggingface.co/albanecoiffe/cnn-image-classifier/resolve/main/model2.pt"

    if not os.path.exists(model_path):
        with st.spinner("📦 Downloading model from Hugging Face..."):
            try:
                response = requests.get(url)
                response.raise_for_status()
                with open(model_path, 'wb') as f:
                    f.write(response.content)
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Error downloading model: {e}")
                raise

    model2 = CNNNet()
    model2.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model2.eval()
    return model2


model = load_model()

# -------------------------------
# Transform and DataLoader
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor()
])

test_dataset = ImageFolder("data/Testing_Data", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
class_names = test_dataset.classes
idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}

# -------------------------------
# Confusion Matrix & Classification Report
# -------------------------------
y_true_list, y_pred_list = [], []
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        y_test_pred = model(x_batch)
        _, y_pred_tag = torch.max(y_test_pred, dim=1)
        y_true_list.extend(y_batch.numpy())
        y_pred_list.extend(y_pred_tag.numpy())

cm = confusion_matrix(y_true_list, y_pred_list)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix")
st.pyplot(fig)

st.subheader("📋 Classification Report")
report = classification_report(y_true_list, y_pred_list, target_names=class_names, output_dict=True)
for cls, scores in report.items():
    if isinstance(scores, dict):
        st.markdown(f"**{cls}** → Precision: `{scores['precision']:.2f}`, Recall: `{scores['recall']:.2f}`, F1: `{scores['f1-score']:.2f}`")


# -------------------------------
# Image selection and prediction
# -------------------------------
st.header("\U0001F5C2️ Select an image from dataset")
image_files = glob.glob("data/Testing_Data/*/*.jpg") + glob.glob("data/Testing_Data/*/*.png")
if len(image_files) > 0:
    selected_image_path = st.selectbox("Choose an image:", image_files)
    image = Image.open(selected_image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    st.image(image, caption="Selected Image", use_column_width=True)

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0].numpy()
        pred_idx = np.argmax(probs)
        pred_label = class_names[pred_idx]

    true_label = selected_image_path.split(os.sep)[-2]
    st.subheader(f"📌 Prediction: **{pred_label}**")
    st.markdown(f"🟩 True label: **{true_label}**")
    st.markdown("### 🔢 Probabilities:")
    for i, prob in enumerate(probs):
        st.markdown(f"{class_names[i]}: `{prob:.4f}`")

