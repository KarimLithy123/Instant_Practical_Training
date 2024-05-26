import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('cupmodel.pt')

results=model(source=0,show=True,conf=0.25,save=True)
