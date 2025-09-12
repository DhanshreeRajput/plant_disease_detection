# app/model_loader.py
import torch
from app.CNN_model import PlantDiseaseCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_path: str = "models/model.pth", num_classes: int = 38):
    model = PlantDiseaseCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model
