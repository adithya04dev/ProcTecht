import torch
from ultralytics import YOLO
# from inference_sdk import InferenceHTTPClient

from models.utils.yolo import parse_yolo_predictions
from models.device import current_device
from config import MIN_FIRE_CONF,ROBOFLOW_API_KEY

fireYoloModel = YOLO(r"C:\Users\adith\Documents\Projects\python-projects\Rakshak-2.0\trained_models\fire_smoke.pt").to(torch.device(current_device))
print("Fire YOLO Loaded")


def detect_fire_dummy(frame):
    return [{'bbox': [0.6313255190849304, 0.430192756652832, 0.244161981344223, 0.39618929624557495],
            'bbox_std': [646.0918579101562, 466.3426513671875, 540.1009521484375, 367.1800842285156],
             'confidence': 0.4033458542823792,
             'label': 'fire',
             'orig_shape': (740, 1216)},
            ]

def detect_fire(frame):
    outputs = fireYoloModel.predict(source=frame, verbose=False)

    # CLIENT = InferenceHTTPClient(
    #         api_url="https://detect.roboflow.com",
    #         api_key=ROBOFLOW_API_KEY
    #     )

    # outputs = CLIENT.infer(frame, model_id="firesmoke-qedaj/5")
    return parse_yolo_predictions(outputs, MIN_FIRE_CONF)
