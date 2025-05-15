import torchvision.transforms as transforms
from PIL import Image
import io
import os

def _get_transform():
    """
    Returns the transformation pipeline for preprocessing images.
    
    The transformation includes resizing to 32x32 pixels, converting to tensor,
    and normalizing with the same values used during model training.
    
    Returns:
        torchvision.transforms.Compose: The transformation pipeline
    """
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])

def process_image(image_input):
    """
    Process an image from bytes or file path to a normalized tensor ready for prediction.
    
    Args:
        image_input (bytes or str): Raw image data or path to image file
        
    Returns:
        torch.Tensor: Processed image tensor with batch dimension added
    """
    # Handle different input types
    if isinstance(image_input, bytes):
        # Input is bytes
        image = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, str) and os.path.isfile(image_input):
        # Input is a file path
        image = Image.open(image_input)
    else:
        raise ValueError("Input must be image bytes or valid file path")
    
    # Apply transformations
    transform = _get_transform()
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor 