from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
import uvicorn

# Import the model and utils
from utils.image_utils import process_image
from utils.model_utils import predict_image, load_model

# Create FastAPI app
app = FastAPI(
    title="CIFAR-10 Image Classifier",
    description="A simple API for classifying images into CIFAR-10 categories",
    version="1.0.0"
)

# Set up templates
templates = Jinja2Templates(directory="templates")

# Load the model at startup
MODEL_PATH = os.path.join("models", "cifar_net.pth")
model = load_model(MODEL_PATH)


# ******************************************************
# Define routes
# ******************************************************



@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    """Serve the main page with the upload form"""
    return templates.TemplateResponse("form.html", {"request": request})



@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Process an uploaded image and return the prediction"""
    # Read the file contents
    contents = await file.read()
    
    # Process the image
    image_tensor = process_image(contents)
    
    # Make prediction
    result = predict_image(model, image_tensor)
    
    # Add filename to result
    result["filename"] = file.filename
    
    return result

if __name__ == "__main__":
    # Start the FastAPI application
    uvicorn.run(app, host="0.0.0.0", port=8000) 