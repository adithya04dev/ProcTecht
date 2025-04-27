import torch
from ultralytics import YOLO
# from inference_sdk import InferenceHTTPClient

from models.device import current_device
from config import MIN_ACCIDENT_CONF,ROBOFLOW_API_KEY

accidentsYoloModel = YOLO(r"C:\Users\adith\Documents\Projects\python-projects\Rakshak-2.0\trained_models\realtime_accident.pt").to(torch.device(current_device))
# def parse_yolo_predictions(predictions, min_conf=0.7):
#     if predictions is None or len(predictions['predictions']) == 0:
#         return []

#     results = []
#     for i in range(len(predictions['predictions'])):
#         prediction = predictions['predictions'][i]
#         print(prediction)   

#         if (prediction['class'] == "car_bike_accident" or prediction['class'] == "car_car_accident" or prediction['class'] == "car_object_accident" or prediction['class'] == "car_person_accident"):

#             if prediction['confidence'] > min_conf:
#                 print(prediction['class'])
#                 print(prediction['confidence'])    


#                 results.append({
#                     'confidence': prediction['confidence'],
#                     'label': prediction['class'],
#                     'bbox': [prediction['x'] / predictions['image']['width'],
#                             prediction['y'] / predictions['image']['height'],
#                             prediction['width'] / predictions['image']['width'],
#                             prediction['height'] / predictions['image']['height']],
#                     'bbox_std': [prediction['x'], prediction['y'], prediction['width'], prediction['height']],
#                     'orig_shape': (predictions['image']['width'], predictions['image']['height'])
#                 })
#     print(results)   
#     return results

def parse_yolo_predictions(predictions, min_conf=0.5):
    if predictions is None or len(predictions) == 0:
        return []

    results = []
    for i in range(len(predictions[0].boxes)):
        if predictions[0].boxes[i].conf[0].item() >= min_conf  and predictions[0].names[predictions[0].boxes[i].cls.item()] in['bike_bike_accident','car_person_accident','car_object_accident','car_car_accident','car_bike_accident','bike_person_accident','bike_object_accident','bike_bike_accident']:
            results.append({
                'confidence': predictions[0].boxes[i].conf[0].item(),
                'label': predictions[0].names[predictions[0].boxes[i].cls.item()],
                'bbox': predictions[0].boxes[i].xywhn[0].tolist(),
                'bbox_std': predictions[0].boxes[i].xywh[0].tolist(),
                'orig_shape': predictions[0].orig_shape
            })
            print(results)
    return results

print("Accidents YOLO Loaded")
def detect_accident_dummy(frame):
    return [{'bbox': [0.5313255190849304, 0.630192756652832, 0.444161981344223, 0.49618929624557495],
            'bbox_std': [646.0918579101562, 466.3426513671875, 540.1009521484375, 367.1800842285156],
             'confidence': 0.9333458542823792,
             'label': 'Accident',
             'orig_shape': (740, 1216)},
            ]



def detect_accident(frame):
    outputs = accidentsYoloModel.predict(source=frame, verbose=False)
    # print(f"Detected {len(outputs)} accidents")
    return parse_yolo_predictions(outputs, MIN_ACCIDENT_CONF)

    # img_path='/content/download.jpg'
    # CLIENT = InferenceHTTPClient(
    #     api_url="https://detect.roboflow.com",
    #     api_key=ROBOFLOW_API_KEY
    # )

    # result = CLIENT.infer(frame, model_id="real-time-accident-detection/1")
    # return parse_yolo_predictions(result, MIN_ACCIDENT_CONF)