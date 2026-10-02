from ultralytics import YOLO

# Load YOLO11 nano segmentation model
model = YOLO("yolo11n-seg.pt")
# Train the model
model.train(
    data="dataset/data.yaml",
    epochs=20,
    imgsz=640,
    batch=4,
    workers=0,
    project="results",
    name="microplastic_seg"
)