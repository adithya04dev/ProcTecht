import torch
from ultralytics import YOLO

from models.utils.yolo import parse_yolo_predictions
from models.device import current_device
from config import MIN_CRACK_CONF,ROBOFLOW_API_KEY
# from inference_sdk import InferenceHTTPClient

crackYoloModel = YOLO(
    r"C:\Users\adith\Documents\Projects\python-projects\Rakshak-2.0\trained_models\cracks_dents.pt").to(torch.device(current_device))
print("Crack YOLO Loaded")


def detect_crack_dummy(frame):
    return [{'bbox': [0.6313255190849304, 0.430192756652832, 0.244161981344223, 0.39618929624557495],
            'bbox_std': [646.0918579101562, 466.3426513671875, 540.1009521484375, 367.1800842285156],
             'confidence': 0.7833458542823792,
             'label': 'fire',
             'orig_shape': (740, 1216)},
            ]

def detect_crack(frame):
    try:
        # CLIENT = InferenceHTTPClient(
        #     api_url="https://outline.roboflow.com",
        #     api_key=ROBOFLOW_API_KEY
        # )

        # result = CLIENT.infer(frame, model_id="crack-and-dent-prznz/2")
        result = crackYoloModel.predict(source=frame, verbose=False)
        parsed_yolo_preds = parse_yolo_predictions(result, MIN_CRACK_CONF)
        # for parsed_yolo_pred in parsed_yolo_preds:
        #     parsed_yolo_pred['label'] = 'crack'
    except Exception as e:
        print("Error in YOLO crack: ", e)
        return []
    return parsed_yolo_preds