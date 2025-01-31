import torch
from ultralytics import YOLO

from models.utils.yolo import parse_yolo_predictions
from models.device import current_device
# from inference_sdk import InferenceHTTPClient
# from config import ROBOFLOW_API_KEY
from config import MIN_WEAPON_CONF
weaponsYoloModel = YOLO(r"C:\Users\adith\Documents\Projects\python-projects\Rakshak-2.0\trained_models\guns_knives.pt").to(torch.device(current_device))
print("Weapons YOLO Loaded")

def detect_weapons_dummy(frame):
    return [{'bbox': [0.5313255190849304, 0.630192756652832, 0.444161981344223, 0.49618929624557495],
            'bbox_std': [646.0918579101562, 466.3426513671875, 540.1009521484375, 367.1800842285156],
             'confidence': 0.9333458542823792,
             'label': 'Gun',
             'orig_shape': (740, 1216)},
            {'bbox': [0.267814576625824, 0.5433048009872437, 0.09092368930578232, 0.708624541759491],
            'bbox_std': [325.66253662109375, 402.0455627441406, 110.56320190429688, 524.3821411132812],
             'confidence': 0.7023518085479736,
             'label': 'Gun',
             'orig_shape': (740, 1216)},
            {'bbox': [0.18045872449874878, 0.5444341897964478, 0.10244493931531906, 0.6700120568275452],
             'bbox_std': [219.43780517578125, 402.88128662109375, 124.57304382324219, 495.8089294433594],
             'confidence': 0.47320428490638733,
             'label': 'Knife',
             'orig_shape': (740, 1216)}]


def detect_weapons(frame):
    result = weaponsYoloModel.predict(source=frame, verbose=False)
    # # print(len(outputs[0].boxes), "weapons, labels => ", outputs[0].names)
    # result = parse_yolo_predictions(outputs)
    # filtered = []
    # for detection in result:
    #     if detection['label'] != 'Missile':
    #         filtered.append(detection)

    # img_path = '/content/download (5).jpg'


    # CLIENT = InferenceHTTPClient(
    #     api_url="https://detect.roboflow.com",
    #     api_key=ROBOFLOW_API_KEY
    # )

    # # execute the method
    # result = CLIENT.infer(frame, model_id="guns_n_knives-h4bky/3")
    

    return parse_yolo_predictions(result,MIN_WEAPON_CONF)
