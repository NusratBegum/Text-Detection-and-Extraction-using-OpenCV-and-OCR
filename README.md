# Text Detection and Extraction using OpenCV and OCR

A deep learning-based Handwriting Recognition system that uses Computer Vision techniques to detect and transcribe handwritten text from images.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [References](#references)
- [License](#license)

---

## Overview

This project implements an end-to-end Optical Character Recognition (OCR) system specifically designed for handwritten text recognition. While traditional OCR systems perform well on machine-printed text, handwritten text recognition remains challenging due to the significant variation in individual writing styles.

### Business Context

Character Recognition technology converts characters on scanned documents into digital forms. This project focuses on recognizing handwritten names, which has applications in:

- **Document Digitization** - Converting handwritten forms to digital text
- **Postal Services** - Reading handwritten addresses
- **Banking** - Processing handwritten checks and forms
- **Healthcare** - Digitizing handwritten medical records

---

## Features

- **Comprehensive EDA** - Detailed exploratory data analysis with visualizations
- **OpenCV Preprocessing** - Advanced image preprocessing pipeline
- **Deep Learning Model** - CNN-based architecture with CTC loss
- **Real-time Inference** - Predict text from new handwritten images
- **Performance Metrics** - CER, WER, and accuracy evaluation

---

## Dataset

**Source:** [Kaggle - Handwriting Recognition Dataset](https://www.kaggle.com/datasets/landlord/handwriting-recognition)

| Split | Samples |
|-------|---------|
| Training | 331,059 |
| Validation | 41,382 |
| Test | 41,382 |
| **Total** | **413,823** |

### Dataset Characteristics

- **Content:** 400,000+ handwritten names collected through charity projects
- **Categories:** 206,799 first names and 207,024 surnames
- **Image Format:** JPG images with variable dimensions
- **Labels:** Text transcriptions of handwritten content

---

## Project Structure

```
Text Detection and Extraction using OpenCV and OCR/
│
├── main.ipynb                          # Main Jupyter notebook with complete pipeline
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
│
├── model_architecture.png              # Model visualization (generated)
├── handwriting_recognition_model.keras # Trained model (generated)
├── model_config.json                   # Model configuration (generated)
├── best_model.keras                    # Best checkpoint (generated)
│
└── data/                               # Dataset directory (downloaded via kagglehub)
    ├── train_v2/
    ├── test_v2/
    ├── validation_v2/
    └── *.csv                           # Label files
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/handwriting-recognition.git
   cd handwriting-recognition
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Kaggle API (for dataset download)**
   - Create a Kaggle account at [kaggle.com](https://www.kaggle.com)
   - Go to Account Settings → API → Create New Token
   - Place `kaggle.json` in `~/.kaggle/` directory

---

## Usage

### Running the Notebook

1. Launch Jupyter Notebook or VS Code
   ```bash
   jupyter notebook main.ipynb
   ```

2. Run cells sequentially from top to bottom

### Quick Start

```python
# Load the trained model
from tensorflow.keras.models import load_model
import json

model = load_model('handwriting_recognition_model.keras')

with open('model_config.json', 'r') as f:
    config = json.load(f)

# Preprocess and predict
img = preprocess_image('path/to/handwritten_image.jpg')
prediction = model.predict(img)
text = decode_prediction(prediction)
print(f"Predicted text: {text}")
```

### Image Preprocessing

```python
import cv2
import numpy as np

def preprocess_image(img_path, target_width=256, target_height=64):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize maintaining aspect ratio
    h, w = img.shape
    aspect_ratio = w / h
    new_width = int(target_height * aspect_ratio)
    
    if new_width > target_width:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
    
    img_resized = cv2.resize(img, (new_width, new_height))
    
    # Center on white canvas
    canvas = np.ones((target_height, target_width), dtype=np.uint8) * 255
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = img_resized
    
    # Normalize
    img_normalized = canvas.astype(np.float32) / 255.0
    return np.expand_dims(img_normalized, axis=-1)
```

---

## Model Architecture

### CRNN (Convolutional Recurrent Neural Network)

```
Input Image (64 x 256 x 1)
         │
         ▼
┌─────────────────────┐
│   CNN Backbone      │  Conv2D (16) → MaxPool
│   Feature Extractor │  Conv2D (32) → MaxPool
│                     │  Conv2D (64) → MaxPool
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Reshape Layer     │  Prepare for sequence processing
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Dense Layers      │  Dense (64) → Dropout (0.2)
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Output Layer      │  Dense (67) + Softmax
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   CTC Loss          │  Connectionist Temporal Classification
└─────────────────────┘
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Input Size | 256 × 64 × 1 |
| Vocabulary Size | 67 characters |
| Max Label Length | 32 characters |
| Optimizer | Adam (lr=0.001) |
| Loss Function | CTC Loss |

---

## Results

### Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **CER** | Character Error Rate | < 15% |
| **WER** | Word Error Rate | < 20% |
| **Accuracy** | Exact Match Accuracy | > 85% |

### Sample Predictions

| Ground Truth | Prediction | Match |
|--------------|------------|-------|
| John | John | ✓ |
| Smith | Smith | ✓ |
| Michael | Micheal | ✗ |

*Note: Results will vary based on training duration and dataset size used.*

---

## Technologies Used

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| TensorFlow | 2.x | Deep learning framework |
| OpenCV | 4.x | Image processing |
| NumPy | 1.x | Numerical computing |
| Pandas | 2.x | Data manipulation |
| Matplotlib | 3.x | Visualization |
| Seaborn | 0.x | Statistical visualization |
| Plotly | 5.x | Interactive plots |

### Deep Learning Components

- **Convolutional Neural Networks (CNN)** - Feature extraction
- **CTC Loss** - Sequence-to-sequence alignment
- **Adam Optimizer** - Gradient descent optimization

---

## Future Improvements

1. **Data Augmentation**
   - Rotation, elastic deformation, noise injection
   - Synthetic data generation

2. **Model Enhancements**
   - Transformer-based architecture (TrOCR)
   - Attention mechanisms
   - EfficientNet backbone

3. **Training Optimization**
   - Curriculum learning
   - Mixed precision training
   - GPU acceleration

4. **Production Deployment**
   - TensorFlow Lite / ONNX optimization
   - REST API with Flask/FastAPI
   - Docker containerization

5. **Post-processing**
   - Spell checking
   - Language model integration
   - Confidence thresholds

---

## References

1. Shi, B., Bai, X., & Yao, C. (2016). *An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition*. IEEE TPAMI.

2. Graves, A., et al. (2006). *Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks*. ICML.

3. [Kaggle Handwriting Recognition Dataset](https://www.kaggle.com/datasets/landlord/handwriting-recognition)

4. [TensorFlow CTC Documentation](https://www.tensorflow.org/api_docs/python/tf/nn/ctc_loss)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Author

**Data Science Team**  
December 2025

---

## Acknowledgments

- Kaggle for hosting the handwriting dataset
- TensorFlow team for the deep learning framework
- OpenCV community for computer vision tools
