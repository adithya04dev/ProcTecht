import platform
import cv2
# from keras.models import load_model
# from keras.optimizers import legacy
import numpy as np
# from inference_sdk import InferenceHTTPClient
from config import ROBOFLOW_API_KEY
import torch
from ultralytics import YOLO
# from inference_sdk import InferenceHTTPClient

from models.utils.yolo import parse_yolo_predictions
from models.device import current_device
from config import MIN_FIRE_CONF,ROBOFLOW_API_KEY
fightYoloModel = YOLO(r"C:\Users\adith\Documents\Projects\python-projects\Rakshak-2.0\trained_models\fights.pt").to(torch.device(current_device))
print("Weapons YOLO Loaded")



def detect_fight(frame):
    result = fightYoloModel.predict(source=frame, verbose=False)
    # CLIENT = InferenceHTTPClient(
    #     api_url="https://detect.roboflow.com",
    #     api_key=ROBOFLOW_API_KEY
    # )

    # result = CLIENT.infer(frame, model_id="gisc/1")
    return parse_yolo_predictions(result)

# def parse_yolo_predictions(predictions, min_conf=0.1):
#     if predictions is None or len(predictions['predictions']) == 0:
#         return []

#     results = []
#     for i in range(len(predictions['predictions'])):
        
#         prediction = predictions['predictions'][i]
#         if prediction['class']=='fight' and prediction['confidence'] >= min_conf:
#             results.append({
#                 'prediction_confidence': prediction['confidence'],
#                 'predicted_class': prediction['class_id'],
#                 'prediction_label': prediction['class'],
#                 'bbox': [prediction['x'] / predictions['image']['width'],
#                          prediction['y'] / predictions['image']['height'],
#                          prediction['width'] / predictions['image']['width'],
#                          prediction['height'] / predictions['image']['height']],
#                 'bbox_std': [prediction['x'], prediction['y'], prediction['width'], prediction['height']],
#                 'orig_shape': (predictions['image']['width'], predictions['image']['height'])
#             })
#     return results


# fightModel = load_model("models/weights/fights/fights2vgg2cat2.keras")
# print("Fight model loaded")

# if platform.system() == "Darwin" and platform.processor() == "arm":
#     # You can adjust the learning rate if needed
#     legacy_adam = legacy.Adam(learning_rate=0.001)
#     fightModel.compile(optimizer=legacy_adam,
#                        loss='binary_crossentropy', metrics=['accuracy'])
#     print("Optimised fight model for apple silicon")


# FIGHT_FRAME_WIDTH = 128
# FIGHT_FRAME_HEIGHT = 128
# # Number of frames per video clip (must match the training configuration)
# FIGHT_NUM_FRAMES = 30
# frame_buffer = []


# def reset_buffer():
#     global frame_buffer
#     frame_buffer = []


# FIGHT_PREDICTION_CLASSES = {
#     0: 'no-fight',
#     1: 'fight'
# }

# dummy_called = 0


# def detect_fight_dummy(frame):
#     global dummy_called
#     dummy_called += 1
#     if dummy_called == FIGHT_NUM_FRAMES:
#         predictions = [[0.9999999, 0.0000001]]
#         return {
#             'prediction_confidence': predictions[0][0],
#             'predicted_class': 0,
#             'prediction_label': 'fight'
#         }
#     return None


# fight_detection = None


# def detect_fight(frame):
#     global fight_detection
#     global frame_buffer

#     standardizedFrame = cv2.resize(
#         frame, (FIGHT_FRAME_WIDTH, FIGHT_FRAME_HEIGHT))
#     frame_buffer.append(standardizedFrame)

#     # Limit to FIGHT_NUM_FRAMES frames
#     if len(frame_buffer) > FIGHT_NUM_FRAMES:
#         frame_buffer = frame_buffer[-7:]

#     # print("Buffer size", len(frame_buffer))

#     if len(frame_buffer) == FIGHT_NUM_FRAMES:
#         input_video_clip = np.array(frame_buffer)
#         input_video_clip = np.expand_dims(input_video_clip, axis=0)

#         # Perform predictions
#         prediction = fightModel.predict(input_video_clip, verbose=False)

#         # Get the class with the highest probability as the predicted class
#         predicted_class = np.argmax(prediction, axis=1)

#         # print("Performed Fight detection: ", prediction)

#         fight_detection = {
#             'predicted_class':  int(predicted_class[0]),
#             'prediction_confidence': float(prediction[0][predicted_class[0]]),
#             'prediction_label': FIGHT_PREDICTION_CLASSES[predicted_class[0]],
#         }
#         # print("Performed fight detection:", fight_detection)
#         reset_buffer()
#     else:
#         pass

#     return fight_detection