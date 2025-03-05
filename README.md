
# Glaucoma Detection with Deep Learning


![Sample Eye Fundus Image](https://github.com/thilak-r/mini-project/blob/master/Screenshot%202024-11-12%20223137.png)

This repository contains a PyTorch implementation of a deep learning model for detecting glaucoma using eye fundus images. The project utilizes a fine-tuned ResNet-18 model, trained on a custom dataset. The application also includes a Flask web interface for uploading images and obtaining predictions.


## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Model Architecture](#model-architecture)
- [Training](#training)
  - [Data Augmentation](#data-augmentation)
  - [Training Script](#training-script)
- [Evaluation](#evaluation)
  - [Confusion Matrix](#confusion-matrix)
  - [ROC Curve](#roc-curve)
- [Web Application](#web-application)
  - [Flask Application](#flask-application)
  - [API Endpoint](#api-endpoint)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)


# Introduction

Glaucoma is a leading cause of irreversible blindness worldwide. Detecting glaucoma early can help in timely treatment and prevent vision loss. This project leverages a deep learning approach using ResNet-18 to classify eye fundus images as either "glaucoma" or "normal".

## Dataset

The dataset used for this project is a collection of fundus images stored in the following structure:

The dataset is split into training (70%), validation (15%), and testing (15%) sets.

## Requirements

Dependencies:

- Python 3.8+
- PyTorch
- Torchvision
- Matplotlib
- Seaborn
- Flask
- Pillow
- scikit-learn
- TQDM

## Model Architecture

The model uses ResNet-18 pretrained on ImageNet, with the fully connected layer replaced by:

- `Linear(model.fc.in_features, 512)`
- `ReLU()`
- `Dropout(0.4)`
- `Linear(512, 2)`

The final output layer consists of two classes: "glaucoma" and "normal".

## Training

The training process uses the following configuration:

- Loss Function: CrossEntropyLoss
- Optimizer: Adam with learning rate 0.001
- Learning Rate Scheduler: ReduceLROnPlateau
- Early Stopping: Patience of 5 epochs

### Data Augmentation

The data is augmented with the following transformations:

- Resize to 224x224
- Random horizontal flip
- Random rotation (±15 degrees)
- Normalization with mean [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225]

### Training Script

To train the model, run:

```bash
python train.py
```

The training history is stored in a dictionary and the best model is saved as `best_model.pth`.

## Evaluation

The model is evaluated on the test set using various metrics, including accuracy, confusion matrix, and ROC-AUC score.

### Confusion Matrix

The confusion matrix visualizes the performance of the model:

### ROC Curve

The ROC curve is plotted to show the trade-off between the true positive rate and false positive rate:

## Web Application

A Flask web application is provided for real-time predictions. Users can upload an image of an eye fundus, and the model will classify it as either "glaucoma" or "normal".

### Flask Application

To run the Flask app locally, use:

```bash
python app.py
```

Navigate to `http://127.0.0.1:5000/` in your browser to use the application.

### API Endpoint

**POST /predict**:
- **Input**: Image file (png, jpg, jpeg)
- **Output**: JSON with prediction (glaucoma or normal) and confidence score

Example request:

```bash
curl -X POST -F "image=@test_image.jpg" http://127.0.0.1:5000/predict
```

Example response:

```json
{
  "prediction": "glaucoma",
  "score": 0.95
}
```

## Results

The model achieved the following performance metrics:

- Training Accuracy: 99.97%
- Validation Accuracy: 99.98%
- Test Accuracy: 99.8%

The model also demonstrated high sensitivity and specificity, as seen in the ROC curve.

## Installation

Clone this repository and navigate into the project directory:

```bash
git clone https://github.com/thilak-r/glaucoma-detection.git
cd glaucoma-detection
```

## Usage

**Train the Model:**

```bash
python train.py
```

**Evaluate the Model:**

```bash
python evaluate.py
```

**Run the Flask App:**

```bash
python app.py
```

Upload a test image and view the prediction.


To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## Acknowledgements

under guidance of agughasi victor [https://github.com/Victor-Ikechukwu](https://github.com/Victor-Ikechukwu)
