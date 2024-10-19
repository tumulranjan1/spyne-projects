from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from torchvision import transforms
import io

def load_trained_model(model_path, device):
    model = torch.jit.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model

app = FastAPI()

# Load the model (replace with your saved model path)
model_path = './bestmodel.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_trained_model(model_path, device)

# Define class names (ensure these match what you used during training)
class_names = ['0', '40', '90', '130', '180', '230', '270', '320']

# Define the transform for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Prediction endpoint
@app.post("/predict/")
async def predict_angle(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        confidence = torch.softmax(output, dim=1)[0][predicted].item()
    
    angle_class = class_names[predicted.item()]  # Get the class label instead of index
    
    return {"angle_class": angle_class, "confidence": confidence}

# FastAPI will automatically provide Swagger UI at /docs
