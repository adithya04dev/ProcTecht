import torch
from ultralytics import YOLO

# from models.utils.yolo import parse_yolo_predictions
from models.device import current_device
def parse_yolo_predictions(predictions, min_conf=0.5):
    if predictions is None or len(predictions) == 0:
        return []

    results = []
    for i in range(len(predictions[0].boxes)):
        if predictions[0].boxes[i].conf[0].item() >= min_conf:
            results.append({
                'confidence': predictions[0].boxes[i].conf[0].item(),
                'label': predictions[0].names[predictions[0].boxes[i].cls.item()],
                'bbox': predictions[0].boxes[i].xywhn[0].tolist(),
                'bbox_std': predictions[0].boxes[i].xywh[0].tolist(),
                'orig_shape': predictions[0].orig_shape
            })
    return results
pretrainedYoloModel = YOLO("models/weights/yolo_v8/yolov8n.pt").to(torch.device(current_device))
print("Pre-trained YOLO Loaded")

def detect_objects_dummy(frame):
    return [
        {'label': 'person', 'confidence': 0.95, 'bbox': [10, 10, 50, 50]},
        {'label': 'car', 'confidence': 0.95, 'bbox': [10, 10, 50, 50]},
        {'label': 'computer', 'confidence': 0.95, 'bbox': [10, 10, 50, 50]}
    ]

def detect_objects(frame):
    try:
        outputs = pretrainedYoloModel.predict(source=frame, verbose=False)
    except Exception as e:
        print("Error in YOLO detection: ", e)
        return []
    return parse_yolo_predictions(outputs, 0.2)
