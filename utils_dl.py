# utils_dl.py

import os
import pickle
from datetime import datetime
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image

ENCODINGS_PATH = "encodings_dl.pkl"
ATTENDANCE_PATH = "attendance_dl.pkl"

# --------- Load / Save Encodings ---------
def load_encodings(path=ENCODINGS_PATH):
    if not os.path.exists(path):
        return [], [], []
    with open(path, "rb") as f:
        data = pickle.load(f)
        encodings = data.get("encodings", [])
        names = data.get("names", [])
        positions = data.get("positions", [])
        return encodings, names, positions

def save_encodings(path, encodings, names, positions):
    data = {"encodings": encodings, "names": names, "positions": positions}
    with open(path, "wb") as f:
        pickle.dump(data, f)

# --------- Load / Save Attendance ---------
def load_attendance(path=ATTENDANCE_PATH):
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        data = pickle.load(f)
        # Chuyển datetime từ string về datetime object nếu cần
        for k, records in data.items():
            for i in range(len(records)):
                in_time, out_time = records[i]
                if isinstance(in_time, str):
                    in_time = datetime.fromisoformat(in_time)
                if isinstance(out_time, str) and out_time != "None":
                    out_time = datetime.fromisoformat(out_time)
                else:
                    out_time = None
                records[i] = [in_time, out_time]
        return data

def save_attendance(attendance, path=ATTENDANCE_PATH):
    data_to_save = {}
    for k, records in attendance.items():
        rec_list = []
        for in_time, out_time in records:
            rec_list.append([in_time.isoformat(), out_time.isoformat() if out_time else None])
        data_to_save[k] = rec_list
    with open(path, "wb") as f:
        pickle.dump(data_to_save, f)

# --------- Get Embedding DL ---------
def get_embedding(pil_image, model):
    """
    Chuyển ảnh PIL sang embedding vector bằng model DL.
    - pil_image: PIL.Image RGB
    - model: InceptionResnetV1 đã pretrained
    Trả về: np.array embedding 512-d
    """
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    img_tensor = torch.tensor(np.array(pil_image)).permute(2,0,1).float() / 255.0  # HWC -> CHW
    img_tensor = img_tensor.unsqueeze(0)  # thêm batch dim
    with torch.no_grad():
        embedding = model(img_tensor)
    return embedding.squeeze().numpy()
