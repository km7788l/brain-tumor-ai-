from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.models import load_model

app = FastAPI(title="Brain Tumor Classification API")

# Load the model
try:
    model = load_model("tumor_model.h5")
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

IMAGE_SIZE = 150
labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']

@app.get("/")
def root():
    return {"message": "Brain Tumor Classification API is live."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data))
        opencv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        resized_img = cv2.resize(opencv_img, (IMAGE_SIZE, IMAGE_SIZE))
        reshaped_img = resized_img.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)

        prediction = model.predict(reshaped_img)
        class_idx = np.argmax(prediction, axis=1)[0]
        label = labels[class_idx]
        confidence = float(np.max(prediction[0]))

        return JSONResponse({
            "status": "success",
            "predicted_class": label,
            "confidence": confidence
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
