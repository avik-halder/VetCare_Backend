from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import time
import threading
from pymongo import MongoClient
from datetime import datetime, timedelta
from bson import ObjectId
import pymongo
import joblib
import xgboost as xgb

import sys
print("Python version:", sys.version)

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Atlas connection
client = MongoClient("mongodb+srv://avik:avik1234@cluster0.yrfckol.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["VetCare"]
collection = db["Skin"]
sensor_collection = db["SensorData"]


# Load obj detection and image classifier models
yolo_model = YOLO("yolo11n.pt")
lumpy_model = load_model("lumpy_model.h5")

# Load trained XGBoost Booster
xgb_model = xgb.Booster()
xgb_model.load_model("xgb_model.json")

# Load scaler and label encoder
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# Shared video frame
latest_frame = None
lock = threading.Lock()

# Time tracker for 1-minute MongoDB insert
last_insert_time = datetime.min

@app.post("/upload-frame")
async def upload_frame(file: UploadFile = File(...)):
    global latest_frame, last_insert_time

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = yolo_model.predict(source=frame, conf=0.3, verbose=False)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = yolo_model.names[cls_id]

            if label == "cow":
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                resized = cv2.resize(crop, (224, 224))
                img_arr = resized.astype("float32") / 255.0
                img_arr = np.expand_dims(img_arr, axis=0)

                prediction = lumpy_model.predict(img_arr, verbose=0)[0][0]
                label = "lumpy skin cow" if prediction > 0.5 else "normal cow"

                # Insert into MongoDB once every 1 minute
                now = datetime.now()
                if (now - last_insert_time) > timedelta(minutes=1):
                    collection.insert_one({
                        "timestamp": now,
                        "result": label
                    })
                    last_insert_time = now

            # Draw detection box and label
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            box_color = (255, 0, 0) if "cow" in label else (100, 255, 100)
            text_color = (255, 255, 255)

            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), box_color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5), font, font_scale, text_color, thickness)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    with lock:
        latest_frame = frame.copy()

    return {"status": "Frame received and processed"}

def serialize_doc(doc):
    doc['_id'] = str(doc['_id'])
    if 'timestamp' in doc and hasattr(doc['timestamp'], 'isoformat'):
        doc['timestamp'] = doc['timestamp'].isoformat()
    return doc

@app.get("/logs")
def get_logs():
    logs = list(collection.find().sort("timestamp", pymongo.DESCENDING).limit(50))
    return JSONResponse(content=[serialize_doc(doc) for doc in logs])



@app.get("/video_feed")
def video_feed():
    def stream():
        while True:
            if latest_frame is not None:
                with lock:
                    _, buffer = cv2.imencode(".jpg", latest_frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                       buffer.tobytes() + b'\r\n')
            time.sleep(0.05)  # ~20 FPS
    return StreamingResponse(stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/skin-latest")
def skin_latest():
    try:
        latest_skin_doc = collection.find_one(sort=[("timestamp", pymongo.DESCENDING)])
        if not latest_skin_doc:
            return {"error": "No skin prediction data found."}
        
        return {
            "result": latest_skin_doc["result"],
            "timestamp": latest_skin_doc["timestamp"].isoformat()
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def index():
    return HTMLResponse("""
        <h2>Live Cow Detection</h2>
        <img src="/video_feed" width="800">
    """)


@app.post("/store-sensor")
async def store_sensor(data: dict):
    try:
        accel_mag = (data["Accel_X"]**2 + data["Accel_Y"]**2 + data["Accel_Z"]**2) ** 0.5
        gyro_mag = (data["Gyro_X"]**2 + data["Gyro_Y"]**2 + data["Gyro_Z"]**2) ** 0.5

        data_to_store = {
            "timestamp": datetime.now(),
            "Temp": data["Temp"],
            "MQ": data["MQ"],
            "Accel_X": data["Accel_X"],
            "Accel_Y": data["Accel_Y"],
            "Accel_Z": data["Accel_Z"],
            "Gyro_X": data["Gyro_X"],
            "Gyro_Y": data["Gyro_Y"],
            "Gyro_Z": data["Gyro_Z"],
            "pH": data["pH"],
            "Pulse": data["Pulse"],
            "Accel_Mag": accel_mag,
            "Gyro_Mag": gyro_mag
        }

        sensor_collection.insert_one(data_to_store)
        return {"status": "Sensor data stored successfully"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/sensor-latest")
def get_latest_sensor_data():
    try:
        latest_docs = list(sensor_collection.find().sort("timestamp", pymongo.DESCENDING).limit(10))
        result = []

        for doc in latest_docs:
            doc['_id'] = str(doc['_id'])
            if 'timestamp' in doc and hasattr(doc['timestamp'], 'isoformat'):
                doc['timestamp'] = doc['timestamp'].isoformat()
            result.append(doc)

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/predict-latest")
def predict_latest():
    try:
        latest_doc = sensor_collection.find_one(sort=[("timestamp", pymongo.DESCENDING)])
        if not latest_doc:
            return {"error": "No sensor data found."}

        features = [
            latest_doc["Temp"],
            latest_doc["MQ"],
            latest_doc["Accel_X"],
            latest_doc["Accel_Y"],
            latest_doc["Accel_Z"],
            latest_doc["Gyro_X"],
            latest_doc["Gyro_Y"],
            latest_doc["Gyro_Z"],
            latest_doc["pH"],
            latest_doc["Pulse"],
            latest_doc["Accel_Mag"],
            latest_doc["Gyro_Mag"]
        ]

        X_scaled = scaler.transform([features])
        dmatrix = xgb.DMatrix(X_scaled)

        probs = xgb_model.predict(dmatrix)
        predicted_class = int(np.argmax(probs, axis=1)[0])
        predicted_disease = le.inverse_transform([predicted_class])[0]

        return {
            "prediction": predicted_disease,
            "data_timestamp": latest_doc["timestamp"].isoformat()
        }

    except Exception as e:
        return {"error": str(e)}




from fastapi import Form

@app.post("/predict-image/")
async def predict_image_api(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Could not decode image."}

        resized = cv2.resize(img, (224, 224))
        img_arr = resized.astype("float32") / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)

        prediction = lumpy_model.predict(img_arr, verbose=0)[0][0]
        label = "lumpy skin cow" if prediction > 0.5 else "normal cow"

        return {
            "prediction": label,
            "confidence": float(prediction)
        }

    except Exception as e:
        return {"error": str(e)}
