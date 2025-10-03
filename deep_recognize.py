import os
import pickle
import datetime
import numpy as np
from utils import load_encodings, get_embedding
from facenet_pytorch import InceptionResnetV1
from PIL import Image

ENCODINGS_PATH = "encodings_dl.pkl"
ATTENDANCE_PATH = "attendance_dl.pkl"

model = InceptionResnetV1(pretrained='vggface2').eval()

def load_attendance():
    if os.path.exists(ATTENDANCE_PATH):
        with open(ATTENDANCE_PATH, 'rb') as f:
            return pickle.load(f)
    return {}

def save_attendance(attendance):
    with open(ATTENDANCE_PATH, 'wb') as f:
        pickle.dump(attendance, f)

def recognize_face(img_pil, threshold=0.8):
    encodings, names = load_encodings(ENCODINGS_PATH)
    if not encodings:
        return "Unknown"

    emb = get_embedding(img_pil, model)
    dists = [np.linalg.norm(emb - e) for e in encodings]
    min_idx = np.argmin(dists)
    if dists[min_idx] < threshold:
        # check-in/out
        attendance = load_attendance()
        name = names[min_idx]
        now = datetime.datetime.now()
        if name not in attendance:
            attendance[name] = []
        attendance[name].append(now.strftime("%Y-%m-%d %H:%M:%S"))
        attendance[name] = attendance[name][-10:]
        save_attendance(attendance)
        return name
    else:
        return "Unknown"
