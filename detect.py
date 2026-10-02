from ultralytics import YOLO
import os

# Load trained microplastic model
model = YOLO("model/best.pt")

# Folder containing test images
input_folder = "test_images"

# Folder where detected images will be saved
output_folder = "results/detection"

os.makedirs(output_folder, exist_ok=True)

# Run detection/segmentation
results = model.predict(
    source=input_folder,
    save=True,
    save_txt=True,
    conf=0.25,
    project="results",
    name="detection"
)

print("\nDetection completed successfully!")
print("Results saved in:", output_folder)

# Display detected objects
for result in results:
    if result.boxes is not None:
        count = len(result.boxes)
        print("Particles detected:", count)

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            print(
                f"  {class_name} - "
                f"{confidence * 100:.2f}% confidence"
            )