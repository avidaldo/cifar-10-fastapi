import torchvision.transforms as transforms
from PIL import Image
import io

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

def process_image(image_bytes):
    """
    Process an image from bytes to a normalized tensor ready for prediction.
    
    Args:
        image_bytes (bytes): Raw image data
        
    Returns:
        torch.Tensor: Processed image tensor with batch dimension added
    """
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Apply transformations
    transform = _get_transform()
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor 