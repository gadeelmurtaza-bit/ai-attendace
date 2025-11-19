import os
import pickle
from PIL import Image
import numpy as np
import face_recognition
import pandas as pd
import zipfile, io, csv
from datetime import datetime

ASSETS_DIR = "assets"
ENCODING_FILE = "encodings.pkl"

os.makedirs(ASSETS_DIR, exist_ok=True)

def image_to_encoding(img):
    rgb = np.array(img.convert("RGB"))
    encs = face_recognition.face_encodings(rgb)
    return encs[0] if len(encs) else None


def load_encodings():
    if not os.path.exists(ENCODING_FILE):
        return {}
    return pickle.load(open(ENCODING_FILE, "rb"))


def save_encodings(data):
    pickle.dump(data, open(ENCODING_FILE, "wb"))


def register_student(name, roll, img):
    enc = image_to_encoding(img)
    if enc is None:
        return False, "No face detected"

    path = os.path.join(ASSETS_DIR, f"{roll}.jpg")
    img.save(path)

    data = load_encodings()
    data[roll] = {"name": name, "photo": path, "encoding": enc}
    save_encodings(data)
    return True, "Registered"


def register_bulk(zip_bytes):
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    if "mapping.csv" not in zf.namelist():
        return False, "mapping.csv missing"

    mapping = pd.read_csv(io.BytesIO(zf.read("mapping.csv")))
    data = load_encodings()

    results = []

    for _, row in mapping.iterrows():
        file = row["filename"]
        name = row["name"]
        roll = str(row["roll"])

        if file not in zf.namelist():
            results.append([roll, False, "file missing"])
            continue

        img = Image.open(io.BytesIO(zf.read(file))).convert("RGB")
        ok, msg = register_student(name, roll, img)
        results.append([roll, ok, msg])

    return True, results


def find_best_match(enc, tolerance=0.55):
    data = load_encodings()
    if not data:
        return None, None

    rolls = list(data.keys())
    stored_encs = [data[r]["encoding"] for r in rolls]

    matches = face_recognition.compare_faces(stored_encs, enc, tolerance)
    distances = face_recognition.face_distance(stored_encs, enc)

    best_idx = None
    best_dist = 999

    for i, is_match in enumerate(matches):
        if is_match and distances[i] < best_dist:
            best_idx = i
            best_dist = distances[i]

    if best_idx is None:
        return None, None

    roll = rolls[best_idx]
    return roll, best_dist


def mark_attendance(roll, name):
    today = datetime.now().strftime("%Y-%m-%d")
    file = f"attendance_{today}.csv"
    exists = os.path.exists(file)

    with open(file, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["timestamp", "roll", "name"])
        writer.writerow([datetime.now().isoformat(), roll, name])
