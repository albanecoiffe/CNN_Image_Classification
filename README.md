# 🧠 CNN Image Classification with PyTorch

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-1.12+-red.svg)
![Streamlit](https://img.shields.io/badge/streamlit-app-green.svg)

This project demonstrates how to use a **Convolutional Neural Network (CNN)** built with **PyTorch** to classify images of:
- **Social Security cards**
- **Driving Licenses**
- **Others**

It includes:
- A complete training pipeline in PyTorch
- Model evaluation with confusion matrix & classification report
- A fully interactive **Streamlit web app**

---

## 🖼️ Streamlit Interface

[Streamlit Page](https://cnn-classification-images.streamlit.app/)

> The app lets you select any test image, see the prediction, and view full model evaluation.

---

## 📁 Dataset

Images are organized in a folder structure compatible with `ImageFolder`:

```
data/
🔹 Training_Data/
🔹🔹 driving_license/
🔹🔹 social_security/
🔹🔹 others/
🔹 Testing_Data/
🔹🔹 driving_license/
🔹🔹 social_security/
🔹🔹 others/
```
Each class contains JPEG and PNG images of varied sizes.

⚠️ Note: Due to dataset size limits on GitHub, the repository only contains a sample of the test set (Testing_Data/) with 29 images per class for demonstration purposes. The full training set was used locally to train the model, but is not included in the repository.
---

## 🧠 Model Architecture

The CNN includes:

- 4 convolutional blocks (Conv2D + BatchNorm + ReLU + MaxPool)
- Fully connected layer with dropout
- Output layer for 3-class softmax

Input images are resized to **200x200** and normalized via `ToTensor()`.

```python
self.linear_layers = nn.Sequential(
    nn.Flatten(),
    nn.Linear(256 * 12 * 12, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 3)
)
```

---

## 📊 Results

The app shows:
- ✅ True label
- ✅ Predicted label
- ✅ Class probabilities
- ✅ Confusion Matrix
- ✅ Classification Report (Precision / Recall / F1)
