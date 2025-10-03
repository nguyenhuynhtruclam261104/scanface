import os
from utils import load_encodings, save_encodings, get_embedding
from facenet_pytorch import InceptionResnetV1
from PIL import Image

ENCODINGS_PATH = "encodings_dl.pkl"
DATASET_PATH = "dataset"

# Load model pretrained
model = InceptionResnetV1(pretrained='vggface2').eval()

def enroll_employee(name):
    person_dir = os.path.join(DATASET_PATH, name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)
    # Giả lập: copy 1 số ảnh từ folder hoặc từ webcam
    # Lâm Tổng có thể tích hợp webcam capture ở đây
    encodings, names = load_encodings(ENCODINGS_PATH)
    for img_file in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_file)
        img = Image.open(img_path).convert('RGB')
        emb = get_embedding(img, model)
        encodings.append(emb)
        names.append(name)
    save_encodings(ENCODINGS_PATH, encodings, names)
    print(f"[DL ENROLL] {name} thành công!")
