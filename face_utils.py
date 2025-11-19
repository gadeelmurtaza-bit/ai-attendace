import os
import pickle
from deepface import DeepFace
from PIL import Image
import numpy as np
from datetime import datetime
import pandas as pd

ASSETS = "assets"
os.makedirs(ASSETS, exist_ok=True)

ENC_FILE = "encodings.pkl"


# ------------ ENCODING ------------
def get_embedding(pil):
    img = np.array(pil.convert("RGB"))
    try:
        emb = DeepFace.represent(img, model_name="Facenet512", detector_backend="opencv")
        return np.array(emb[0]["embedding"])
    except:
        return None


# ------------ SAVE/LOAD DATA ------------
def load_data():
    if not os.path.exists(ENC_FILE):
        return {}
    return pickle.load(open(ENC_FILE, "rb"))


def save_data(data):
    pickle.dump(data, open(ENC_FILE, "wb"))


# ------------ REGISTER SINGLE ------------
def register_student(name, roll, pil):
    emb = get_embedding(pil)
    if emb is None:
        return False, "No face detected"

    path = f"{ASSETS}/{roll}.jpg"
    pil.save(path)

    data = load_data()
    data[roll] = {"name": name, "photo": path, "embedding": emb}
    save_data(data)

    return True, "Registered Successfully"


# ------------ MATCH FACE ------------
def match_face(emb):
    data = load_data()
    if not data:
        return None, None

    best_roll = None
    best_sim = -1

    for roll, info in data.items():
        stored = np.array(info["embedding"])

        sim = DeepFace.cosine_similarity(emb, stored)

        if sim > best_sim:
            best_sim = sim
            best_roll = roll

    if best_sim < 0.7:   # threshold
        return None, None

    return best_roll, best_sim


# ------------ ATTENDANCE ------------
def mark_attendance(roll, name):
    today = datetime.now().strftime("%Y-%m-%d")
    file = f"attendance_{today}.csv"

    exists = os.path.exists(file)

    import csv
    with open(file, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["timestamp", "roll", "name"])
        writer.writerow([datetime.now().isoformat(), roll, name])
