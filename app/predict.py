# app/predict.py
import torch
import torchvision.transforms as transforms
from PIL import Image
from app.model_loader import load_model, DEVICE

# Example: Replace with your actual class names
CLASS_NAMES = ["Apple___healthy", "Apple___scab", "Tomato___Late_blight", "Corn___Common_rust"]

transform = transforms.Compose([
    transforms.Resize((64, 64)),   # match training size
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

model = load_model()

def predict_image(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        return CLASS_NAMES[predicted.item()]
