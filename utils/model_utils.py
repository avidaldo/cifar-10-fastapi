import torch
import torch.nn.functional as F
import os
from models.cnn_model import ImprovedCNN

# List of CIFAR-10 classes
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

def load_model(model_path):
    """
    Load a pre-trained PyTorch model.
    
    Args:
        model_path (str): Path to the saved model file
        
    Returns:
        torch.nn.Module: The loaded model in evaluation mode
    """
    # Create a new model instance
    model = ImprovedCNN()
    
    # Check if the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the state dictionary
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    
    # Set the model to evaluation mode
    model.eval()
    
    return model


def predict_image(model, image_tensor):
    """
    Make a prediction on a processed image tensor.
    
    Args:
        model (torch.nn.Module): The trained model
        image_tensor (torch.Tensor): Processed image tensor
        
    Returns:
        dict: Dictionary containing prediction class and confidence
    """
    with torch.no_grad():
        # Get model outputs
        outputs = model(image_tensor)
        
        # Get predicted class index
        _, predicted = torch.max(outputs, 1)
        
        # Calculate softmax probabilities
        probs = F.softmax(outputs, dim=1)
        
    # Get the predicted class and confidence
    class_idx = predicted.item()
    confidence = probs[0][class_idx].item()
    
    return {
        "prediction": CLASSES[class_idx],
        "confidence": round(confidence * 100, 2)
    } 