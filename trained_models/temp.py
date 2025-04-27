from ultralytics import YOLO
import requests
from PIL import Image
from io import BytesIO
#generates random number
from random import randint
def predict(image_url,type):
    # Load the trained model
    
    model = YOLO(f'./trained_models/model_weights/{type}.pt').to('cuda')
    #for local image
    # img = Image.open('./images/gun.png')
    # Fetch the image from the URL
    response = requests.get(image_url)
    img=Image.open(BytesIO(response.content))
    
    # Perform prediction
    results = model(img)
    
    # Display or return results
    for result in results:
        boxes = result.boxes
        masks = result.masks  
        keypoints = result.keypoints
        probs = result.probs 
        obb = result.obb 
        # result.show() 
        result.save(filename=f"./trained_models/result_{type}_{randint(1,100)}.jpg")

# Example usage
image_url = 'https://media.istockphoto.com/id/108224113/photo/agent-007.jpg?s=612x612&w=0&k=20&c=zD78DoadOUbfZgPpyXUXiLscsAVTsBujidCoo4btvDM='
predict_accident(image_url,'guns_knives')
# import torch
# print(f"CUDA available: {torch.cuda.is_available()}")
# print(f"Current device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
# import torch
# print("CUDA Version:", torch.__version__)
# print("GPU Available:", torch.cuda.device_count() > 0)