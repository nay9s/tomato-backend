from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

model = YOLO("best.pt")


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "Tomato AI Backend"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()

    image_array = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Invalid image"}

    result = model.predict(
        image,
        conf=0.25,
        imgsz=640,
        verbose=False
    )[0]

    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "class_id": class_id,
            "disease": result.names[class_id],
            "confidence": round(confidence, 4),
            "bbox": {
                "x1": round(x1),
                "y1": round(y1),
                "x2": round(x2),
                "y2": round(y2)
            }
        })

    return {
        "count": len(detections),
        "detections": detections
    }