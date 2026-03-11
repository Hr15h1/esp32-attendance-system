from flask import Flask, request, jsonify
import numpy as np
import cv2
import matplotlib.pyplot as plt
import joblib
import os
from deepface import DeepFace
from datetime import datetime
import pandas as pd



app = Flask(__name__)


def load_models(app):
    app.config["PCA_MODEL"] = joblib.load("pca_model.pkl")
    app.config("KNN_MODEL") = joblib.load("knn_model.pkl")
    app.config["CLASSES"] = os.listdir("train")

def extract_faces(img):
    
    face_obj = DeepFace.extract_faces(img_path=img, detector_backend="ssd", align = False, normalize_face=False, enforce_detection=False)

    if len(face_obj) == 0:
        return None
    
    face = face_obj[0]["face"]
    face = (face * 255).astype("uint8")

    return face



def preprocess_face(img):
    face = cv2.resize(face, (128, 128))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face_vector = face.flatten() / 255.0

    return face_vector

def predict_face(face_vector, classes):

    pca = app.config["PCA_MODEL"]
    face_pca = pca.transform(face_vector.reshape(1, -1))

    knn_model = app.config["KNN_MODEL"]
    pred = knn_model.predict(face_pca)[0]

    name = classes[pred]

    return name

def mark_attendance(name):
    current_date = str(datetime.now().date())
    current_time = datetime.now().strftime("%H:%M:%S")

    if os.path.exists("attendance.csv"):
        att = pd.read_csv("attendance.csv")
    else:
        att = pd.DataFrame(columns = ["date", "time", "name", "attendance"])

    already_marked = (
        (att["name"] == name & att["date"] == current_date)
    ).any()

    if already_marked:
        return "Attendance already marked."
    
    new_row = {
        "date": current_date,
        "time": current_time,
        "name": name,
        "attendance": "present"
    }

    att.loc[len[att]] = new_row
    att.to_csv("attendance.csv", index = False)

    return "Attendance Marked for student"


@app.route("/upload", methods=["POST"])
def receive_image():

    file = request.files["image"]

    img_bytes = np.frombuffer(file.read(), np.uint8)

    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    print("image received")
    plt.figure(figsize=(5,5))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Image from esp32cam")
    plt.show()  

    return jsonify({
        "status": "received"
    })

if __name__ == "main":
    app.run(host = "0.0.0.0", port=5000)