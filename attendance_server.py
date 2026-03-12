from flask import Flask, request, jsonify
import numpy as np
import cv2
import matplotlib.pyplot as plt
import joblib
import os
from deepface import DeepFace
from datetime import datetime, time
import pandas as pd


# slot1_start = time(9, 10)
# slot1_end = time(9, 10)


# slot2_start = time(10, 0)
# slot2_end = time(10, 10)


# slot3_start = time(11, 0)
# slot3_end = time(11, 10)


# slot4_start = time(13, 0)
# slot4_end = time(13, 10)


# slot5_start = time(14, 0)
# slot5_end = time(14, 10)


# slot6_start = time(15, 0)
# slot6_end = time(15, 10)

timeslots = {
    "Hour 1": (time(9, 0), time(9, 10)),
    "Hour 2": (time(10, 0), time(10, 10)),
    "Hour 3": (time(11, 0), time(11, 10)),
    "Hour 4": (time(13, 0), time(13, 10)),
    "Hour 5": (time(14, 0), time(14, 10)),
    "Hour 6": (time(15, 0), time(15, 10)),
}



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

def is_within_timeslot(current_time):
    for hour, (start, end) in timeslots.items():

        if start <= current_time <= end:
            return hour
    
    return False

def mark_attendance(name):
    current_date = str(datetime.now().date())
    current_time = datetime.now().strftime("%H:%M:%S")
    now = datetime.now().time()

    hour = is_within_timeslot(now)
    
    if not is_within_timeslot(now):
        return "Cannot mark attendance. You are late!!"
    

    if os.path.exists("attendance.csv"):
        att = pd.read_csv("attendance.csv")
    else:
        att = pd.DataFrame(columns = ["date", "time", "name", "attendance"])

    already_marked = (
        (att["name"] == name) & (att["date"] == current_date)
    ).any()

    if already_marked:
        return "Attendance already marked."
    
    new_row = {
        "date": current_date,
        "time": current_time,
        "name": name,
        "attendance": "present"
    }

    att.loc[len(att)] = new_row
    att.to_csv("attendance.csv", index = False)

    return "Attendance Marked for student"

def mark_absentees():
    current_date = str(datetime.now().date())
    current_time = datetime.now().strftime("%H:%M:%S")


    if os.path.exists("attendance.csv"):
        att = pd.read_csv("attendance.csv")
    else:
        att = pd.DataFrame(columns = ["date", "time", "name", "attendance"])

    present_today = att[att["date"] == current_date]["name"].tolist()

    all_students = app.config["CLASSES"]

    absentees = [s for s in all_students if s not in present_today]

    for student in absentees:

        new_row = {
            "date": current_date,
            "time": current_time,
            "name": student,
            "attendance": "absent"
        }

        att.loc[len(att)] = new_row
    att.to_csv("attendance.csv", index=False)
    return "Absentees marked"

    
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