## ProTecht 

ProTecht is an constant surveillance security solution which detects & identifies various types of incidents(fire,accidents etc) from CCTV streams using YOLOv8 models.

We trained YOLOv8 object detection model on 5 different curom tasks like accident,fire,weapons,smoke,climber detection using roboflow datasets.

A web interface platform is built using React and FastAPI for security personals to monitor the detected incidents from trained model.

## Tech Stack 
React & FAST API (Web Interface)  

MongoDB , AWS S3 (For Storage)   

Ultralytics python package (For training Yolo models)   

## Start frontend
```bash
cd Frontend
npm install
npm run dev
```

## Start backend

```bash
cd Backend
python -m venv venv
.\venv\scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port=5000
```

### Start video streaming server
```bash
cd video-server
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python video_server.py
```
