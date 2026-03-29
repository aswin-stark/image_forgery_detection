# 🖼️ Fake vs Real Image Detector
Live On : https://image-forgery-dt.streamlit.app
A Flask web application that uses a deep learning model to classify images as **Real** or **Fake**.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![License](https://img.shields.io/badge/License-Educational-lightgrey.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Model Details](#model-details)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

This project is a web-based image classifier built with **Flask** and **TensorFlow/Keras**. Users can upload an image through the web interface, and the model will predict whether the image is **Real ✅** or **Fake ❌**.

---

## Project Structure

```
project1/
│
├── app.py               # Main Flask application
├── model.pkl            # Trained Keras model
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignored files
│
└── templates/
    └── index.html       # Frontend HTML page
```

---

## Requirements

- Python 3.8+
- TensorFlow 2.13+
- Flask 2.3+
- NumPy 1.24+
- Pillow 9.5+

---

## Installation

### 1. Clone or download the project

```bash
git clone https://github.com/your-username/project1.git
cd project1
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Start the Flask server

```bash
python app.py
```

### 2. Open your browser

Navigate to:

```
http://127.0.0.1:5000
```

### 3. Upload an image

- Click the upload button on the webpage
- Select any image (`.jpg`, `.jpeg`, `.png`)
- Click **Predict**
- The result will display as either **Real ✅** or **Fake ❌**

---

## API Reference

### `POST /predict`

Accepts an image file and returns a prediction.

**Request**

| Field | Type | Description            |
|-------|------|------------------------|
| file  | File | Image file to classify |

**Response**

```json
{
  "result": "Real ✅"
}
```

or

```json
{
  "result": "Fake ❌"
}
```

**Example using `curl`**

```bash
curl -X POST -F "file=@your_image.jpg" http://127.0.0.1:5000/predict
```

---

## Model Details

| Property      | Value               |
|---------------|---------------------|
| Input Shape   | (224, 224, 3)       |
| Normalization | Pixel values / 255  |
| Output        | Sigmoid probability |
| Threshold     | 0.5                 |
| Format        | Keras `.pkl`        |

The model expects RGB images resized to **224×224 pixels**. Pixel values are normalized to the range `[0, 1]` before inference.

**Preprocessing pipeline:**

```python
img = Image.open(file).convert("RGB")
img = img.resize((224, 224))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 224, 224, 3)
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'tensorflow'`
```bash
pip install tensorflow
# or for CPU-only:
pip install tensorflow-cpu
```

### `ValueError: File format not supported`
Ensure you are using `load_model()` from `tensorflow.keras.models` and **not** `joblib.load()` for Keras models.

```python
# ✅ Correct
from tensorflow.keras.models import load_model
model = load_model("model.pkl")

# ❌ Wrong
import joblib
model = joblib.load("model.pkl")
```

### Shape mismatch error
```
Input 0 with name 'input_layer_1' is incompatible: expected shape=(None, 224, 224, 3), found shape=(1, 49152)
```
Make sure images are resized to `(224, 224)` and **not** flattened before passing to the model.

```python
# ✅ Correct
img = img.resize((224, 224))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

# ❌ Wrong
img = img.resize((128, 128))
img_array = np.array(img).flatten().reshape(1, -1)  # (1, 49152)
```

### oneDNN warnings on startup
These are harmless TensorFlow info messages. To suppress them:

```bash
# Windows
set TF_ENABLE_ONEDNN_OPTS=0

# macOS/Linux
export TF_ENABLE_ONEDNN_OPTS=0
```

---

## License

This project is for educational purposes.
