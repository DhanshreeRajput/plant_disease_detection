# app/main.py
from fastapi import FastAPI, UploadFile, File
import shutil
import os
from app.predict import predict_image

app = FastAPI(title="Plant Disease Detection API")

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "Plant Disease Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_image(file_path)

    os.remove(file_path)  # clean up
    return {"prediction": result}
