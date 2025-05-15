# CIFAR-10 Image Classifier API

A simple FastAPI application for classifying images into CIFAR-10 categories using a PyTorch CNN model.

## Overview

This project demonstrates how to deploy a machine learning model (specifically a CNN for image classification) as a web API. The application allows users to upload images through a web interface and get predictions from the model.

## Project Structure

The project has a simplified structure but keeps the models in a separate directory:

```
├── main.py               # Main application file with API endpoints
├── app
│   ├── models            # Model definitions and weights
│   │   ├── cnn_model.py  # CNN model architecture
│   │   └── cifar_net.pth # Pre-trained model weights
│   ├── templates         # HTML templates
│   │   └── form.html     # Upload form template
│   └── static            # Static files (if any)
├── cifar_net.pth         # Pre-trained model copy in root (fallback)
└── requirements.txt      # Project dependencies
```

## Features

- Simple web interface for uploading images
- Image classification into one of the CIFAR-10 categories:
  - airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Real-time prediction with confidence score

## Key Components

1. **FastAPI Application**: Handles HTTP requests and serves the web interface
2. **CNN Model**: A convolutional neural network trained on the CIFAR-10 dataset
3. **Image Processing**: Utilities to preprocess images for prediction
4. **Web Interface**: Simple HTML form for image upload and viewing results

## How It Works

1. The user uploads an image through the web interface
2. The image is preprocessed (resized to 32x32, normalized, etc.)
3. The preprocessed image is fed into the CNN model
4. The model predicts the most likely class
5. The prediction and confidence score are returned to the user

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd cifar-10-fastapi
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Running the Application

Run the application using:

```
python main.py
```

The application will start and be available at http://localhost:8000

### Using the API

- Open your browser and go to http://localhost:8000
- Upload an image using the form
- The application will process the image and display the predicted class and confidence score

## Understanding the Code

### `main.py`
Contains the FastAPI application, image processing functions, and API endpoints.

### `app/models/cnn_model.py`
Contains the CNN architecture definition used for classification.

This simplified structure maintains code organization while being easy to understand for beginners.

## For ML Students

If you're studying machine learning and want to understand how to deploy your models:

1. **Model Definition**: See how the model is defined in `app/models/cnn_model.py`
2. **Model Loading**: See how the model is loaded in `main.py`
3. **Prediction Pipeline**: See how images are processed and fed into the model
4. **Web Integration**: Learn how to create a simple web interface for your ML models
