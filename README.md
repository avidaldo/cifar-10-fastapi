# CIFAR-10 Image Classifier API

A simple FastAPI application for classifying images into CIFAR-10 categories using a PyTorch CNN model.

## Overview

This project demonstrates how to deploy a machine learning model (specifically a CNN for image classification) as a web API. The application allows users to upload images through a web interface and get predictions from the model.

## Project Structure

The project follows a clean, simple structure for educational purposes:

```
├── main.py               # Main application file with API endpoints
├── models/               # Model definitions and weights
│   ├── cnn_model.py      # CNN model architecture
│   └── cifar_net.pth     # Pre-trained model weights
├── templates/            # HTML templates
│   └── form.html         # Upload form template
├── static/               # Static files
│   ├── css/
│   │   └── style.css     # CSS styles
│   └── js/
│       └── script.js     # JavaScript for image preview
├── utils/                # Utility functions
│   ├── image_utils.py    # Image processing utilities
│   └── model_utils.py    # Model loading and prediction utilities
└── requirements.txt      # Project dependencies
```

## Features

- Simple web interface for uploading images
- Image preview before submission
- Image classification into one of the CIFAR-10 categories:
  - airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Server-side rendering of results with confidence score

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
5. The prediction and confidence score are shown on the same page

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

### Separation of Concerns

The project follows good software engineering practices with a clean separation of concerns:

- **HTML (templates/form.html)**: Structure of the web interface
- **CSS (static/css/style.css)**: Styling of the web interface
- **JavaScript (static/js/script.js)**: Client-side interactivity (image preview)
- **Python (main.py)**: Application logic and API endpoints
- **Python (utils/)**: Reusable utility functions
- **Python (models/)**: Model definition and weights

### Implementation Details

- **main.py**: Sets up FastAPI application, routes, and serves the web interface.
- **utils/image_utils.py**: Contains functions for processing images for the model.
- **utils/model_utils.py**: Handles model loading and prediction functions.
- **models/cnn_model.py**: Defines the CNN architecture used for classification.
