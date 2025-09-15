# backend/detection_model.py
from ultralytics import YOLO
import cv2

from utils.config import YOLO_MODEL_PATH
from utils.logger import logger

# Load YOLOv8 model
model = YOLO(YOLO_MODEL_PATH)

def detect_objects(image_path: str) -> dict:
    """
    Run YOLOv8 detection on an image and return objects with labels and confidence.
    """
    results = model(image_path)  # inference
    detections = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            detections.append({"label": label, "confidence": conf})

    logger.info(f"Detections: {detections}")
    return {"objects": detections}
