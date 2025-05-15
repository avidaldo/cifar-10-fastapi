from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import uvicorn
import shutil
import uuid

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

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create uploads directory if it doesn't exist
os.makedirs("static/uploads", exist_ok=True)

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
async def predict(request: Request, file: UploadFile = File(...)):
    """Process an uploaded image and return the prediction"""
    # Create a unique filename for the uploaded image
    file_extension = file.filename.split(".")[-1]
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join("static", "uploads", unique_filename)
    
    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the image for prediction
    image_tensor = process_image(file_path)
    
    # Make prediction
    result = predict_image(model, image_tensor)
    
    # Add filename and image path to result
    result["filename"] = file.filename
    
    # Return the template with the result and image path
    return templates.TemplateResponse(
        "form.html", 
        {
            "request": request, 
            "result": result,
            "image_path": f"/static/uploads/{unique_filename}"
        }
    )

if __name__ == "__main__":
    # Start the FastAPI application
    uvicorn.run(app, host="0.0.0.0", port=8000) 