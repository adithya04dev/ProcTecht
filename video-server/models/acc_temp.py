import torch
from ultralytics import YOLO
from PIL import Image

if torch.cuda.is_available():
    current_device = 'cuda'
else:
    current_device = 'cpu'
model = YOLO(r"C:\Users\adith\Documents\Projects\python-projects\Rakshak-2.0\trained_models\models_results\detect\fire_smoke_new\weights\best.pt").to(torch.device(current_device))
image_path = r"C:\Users\adith\OneDrive\Pictures\Screenshots\Screenshot 2024-06-02 190555.png"
image = Image.open(image_path)
predictions = model.predict(source=image)
# print(predictions)
results = []
if(len(predictions[0].boxes) > 0):
    for i in range(len(predictions[0].boxes)):
        if predictions[0].boxes[i].conf[0].item() >= 0.05:
            results.append({
                'confidence': predictions[0].boxes[i].conf[0].item(),
                'label': predictions[0].names[predictions[0].boxes[i].cls.item()],
                'bbox': predictions[0].boxes[i].xywhn[0].tolist(),
                'bbox_std': predictions[0].boxes[i].xywh[0].tolist(),
                'orig_shape': predictions[0].orig_shape
            })
    print(results)
else:
    print("No predictions")


