import cv2
import os
import pickle
import numpy as np
from PIL import Image
import urllib.request
from datetime import datetime
import pandas as pd

ASSETS = "assets"
os.makedirs(ASSETS, exist_ok=True)

ENC_FILE = "encodings.pkl"

# ----------- DOWNLOAD MODEL -----------
def download_model(url, path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)

# SFace model
download_model(
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/master/sface/face_recognition_sface_2021dec.onnx",
    "sface.onnx"
)

# Face detector
download_model(
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/master/face_detection_yunet/yunet.onnx",
    "yunet.onnx"
)

detector = cv2.FaceDetectorYN.create("yunet.onnx", "", (320, 320), 0.9, 0.3, 5000)
recognizer = cv2.FaceRecognizerSF.create("sface.onnx", "")


# ----------- ENCODING -----------
def get_encoding(pil):
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    detector.setInputSize((w, h))

    faces = detector.detect(img)
    if faces[1] is None:
        return None

    face = faces[1][0:4]
    aligned = recognizer.alignCrop(img, face)
    feat = recognizer.feature(aligned)
    return feat


def load_encodings():
    if not os.path.exists(ENC_FILE):
        return {}
    return pickle.load(open(ENC_FILE, "rb"))


def save_encodings(data):
    pickle.dump(data, open(ENC_FILE, "wb"))


# ----------- REGISTRATION -----------
def register_student(name, roll, pil):
    enc = get_encoding(pil)
    if enc is None:
        return False, "No face detected"

    path = f"{ASSETS}/{roll}.jpg"
    pil.save(path)

    data = load_encodings()
    data[roll] = {"name": name, "photo": path, "encoding": enc}
    save_encodings(data)

    return True, "Registered"


# ----------- MATCHING -----------
def match_face(enc):
    data = load_encodings()
    if not data:
        return None, None

    best_roll = None
    best_score = -1

    for roll, d in data.items():
        score = recognizer.match(enc, d["encoding"], cv2.FaceRecognizerSF_FR_COSINE)
        if score > best_score:
            best_score = score
            best_roll = roll

    if best_score < 0.5:
        return None, None

    return best_roll, best_score


# ----------- ATTENDANCE -----------
def mark_attendance(roll, name):
    today = datetime.now().strftime("%Y-%m-%d")
    file = f"attendance_{today}.csv"
    exists = os.path.exists(file)

    with open(file, "a", newline="") as f:
        import csv
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["timestamp", "roll", "name"])
        writer.writerow([datetime.now().isoformat(), roll, name])
