import os, threading, cv2, pickle, time
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk, ImageDraw
import numpy as np
from datetime import datetime
from facenet_pytorch import InceptionResnetV1
import torch


model = InceptionResnetV1(pretrained=None).eval()
state_dict = torch.load("models/20180402-114759-vggface2.pt", map_location="cpu")
model = InceptionResnetV1(pretrained=None, classify=False)
model.load_state_dict(state_dict, strict=False)

# ---------- Paths ----------
DATASET_PATH = "dataset_dl"
ENCODINGS_PATH = "encodings_dl.pkl"
ATTENDANCE_PATH = "attendance_dl.pkl"
os.makedirs(DATASET_PATH, exist_ok=True)

# ---------- Load/Save ----------
def load_encodings():
    if not os.path.exists(ENCODINGS_PATH):
        return [], [], []
    with open(ENCODINGS_PATH, "rb") as f:
        data = pickle.load(f)
    return data.get("encodings", []), data.get("names", []), data.get("positions", [])

def save_encodings(encodings, names, positions):
    data = {"encodings": encodings, "names": names, "positions": positions}
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(data, f)

def load_attendance():
    if not os.path.exists(ATTENDANCE_PATH):
        return {}
    with open(ATTENDANCE_PATH, "rb") as f:
        return pickle.load(f)

def save_attendance(att):
    with open(ATTENDANCE_PATH, "wb") as f:
        pickle.dump(att, f)

# ---------- DL Model ----------
model = InceptionResnetV1(pretrained='vggface2').eval()

def get_embedding(img_pil):
    img = img_pil.resize((160,160))
    img = np.array(img)/255.0
    img = np.transpose(img,(2,0,1))
    img = np.expand_dims(img,0)
    img_tensor = torch.tensor(img, dtype=torch.float32)
    with torch.no_grad():
        emb = model(img_tensor).numpy()[0]
    return emb

def compare_embeddings(emb1, emb2, threshold=0.8):
    return np.linalg.norm(emb1-emb2) < threshold

# ---------- GUI ----------
root = tk.Tk()
root.title("Hệ thống Nhận diện & Chấm công DL")
root.geometry("1200x700")
root.configure(bg="#f0f0f0")

camera_running = False

# ---------- Sidebar ----------
menu_frame = tk.Frame(root, bg="#2c3e50", width=200)
menu_frame.pack(side="left", fill="y")

def create_sidebar_button(text, color, command):
    btn = tk.Button(menu_frame, text=text, bg=color, fg="white", font=("Arial",12,"bold"),
                    relief="flat", command=command)
    btn.pack(fill="x", padx=10, pady=10)
    return btn

# ---------- Employee Cards ----------
main_frame = tk.Frame(root, bg="#ecf0f1")
main_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)

search_var = tk.StringVar()
search_entry = tk.Entry(main_frame, textvariable=search_var, font=("Arial",12))
search_entry.pack(side="top", fill="x", padx=10, pady=10)

canvas = tk.Canvas(main_frame, bg="#ecf0f1", highlightthickness=0)
scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
scroll_frame = tk.Frame(canvas, bg="#ecf0f1")
scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.create_window((0,0), window=scroll_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

employee_frames = {}
photo_labels = {}

# ---------- Functions ----------
def refresh_cards(*args):
    for frame in employee_frames.values():
        frame.destroy()
    employee_frames.clear()
    encodings, names, positions = load_encodings()
    attendance_data = load_attendance()
    unique_names = {}
    for n,p in zip(names, positions):
        if n not in unique_names:
            unique_names[n]=p
    for name, pos in unique_names.items():
        if search_var.get().lower() not in name.lower():
            continue
        frame = tk.Frame(scroll_frame, bg="white", relief="raised", bd=2)
        frame.pack(fill="x", padx=10, pady=5)
        employee_frames[name]=frame
        # Image
        img_path = os.path.join(DATASET_PATH,name)
        img_file = None
        if os.path.exists(img_path):
            files = [f for f in os.listdir(img_path) if f.lower().endswith((".jpg",".png"))]
            if files: img_file = os.path.join(img_path, files[0])
        if img_file:
            img = Image.open(img_file).resize((80,80))
        else:
            img = Image.new("RGB",(80,80),"gray")
        mask = Image.new("L",(80,80),0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0,0,80,80),fill=255)
        img.putalpha(mask)
        photo = ImageTk.PhotoImage(img)
        photo_labels[name]=photo
        lbl_img = tk.Label(frame, image=photo, bg="white")
        lbl_img.pack(side="left", padx=10, pady=5)
        # Info
        info_frame = tk.Frame(frame, bg="white")
        info_frame.pack(side="left", fill="x", expand=True)
        total_hours=0.0
        records=attendance_data.get(name,[])
        for rec in records:
            in_t,out_t=rec
            if out_t: total_hours+=(out_t-in_t).total_seconds()/3600
        last10=records[-10:]
        last10_str="\n".join([f"{r[0].strftime('%d/%m %H:%M')} - {r[1].strftime('%d/%m %H:%M') if r[1] else '...'}" for r in last10])
        tk.Label(info_frame,text=f"{name} | {pos}",font=("Arial",12,"bold"),bg="white").pack(anchor="w")
        tk.Label(info_frame,text=f"Tổng giờ: {total_hours:.2f}h",bg="white").pack(anchor="w")
        tk.Label(info_frame,text=f"10 lần gần nhất:\n{last10_str}",bg="white",justify="left").pack(anchor="w")
        # Buttons
        btn_frame = tk.Frame(frame,bg="white")
        btn_frame.pack(side="right", padx=5)
        tk.Button(btn_frame,text="Xóa",bg="#e74c3c",fg="white",command=lambda n=name: delete_employee(n)).pack(pady=2)
        tk.Button(btn_frame,text="Đổi tên",bg="#f39c12",fg="white",command=lambda n=name: rename_employee(n)).pack(pady=2)

def delete_employee(name):
    if messagebox.askyesno("Xác nhận",f"Xóa nhân viên {name}?"):
        encodings,names,positions=load_encodings()
        idxs=[i for i,n in enumerate(names) if n==name]
        for i in sorted(idxs,reverse=True):
            encodings.pop(i)
            names.pop(i)
            positions.pop(i)
        save_encodings(encodings,names,positions)
        att=load_attendance()
        att.pop(name,None)
        save_attendance(att)
        person_dir=os.path.join(DATASET_PATH,name)
        if os.path.exists(person_dir):
            for f in os.listdir(person_dir):
                os.remove(os.path.join(person_dir,f))
            os.rmdir(person_dir)
        refresh_cards()

def rename_employee(name):
    new_name=simpledialog.askstring("Đổi tên",f"Nhập tên mới cho {name}")
    if not new_name: return
    encodings,names,positions=load_encodings()
    for i,n in enumerate(names):
        if n==name: names[i]=new_name
    save_encodings(encodings,names,positions)
    att=load_attendance()
    if name in att:
        att[new_name]=att.pop(name)
        save_attendance(att)
    old_dir=os.path.join(DATASET_PATH,name)
    new_dir=os.path.join(DATASET_PATH,new_name)
    if os.path.exists(old_dir):
        os.rename(old_dir,new_dir)
    refresh_cards()

# ---------- Enroll ----------
def enroll_callback():
    popup = tk.Toplevel(root)
    popup.title("Enroll nhân viên mới")
    popup.geometry("450x650")
    popup.grab_set()

    # Hướng dẫn
    tk.Label(popup, text="Hướng dẫn:", font=("Arial",12,"bold")).pack(pady=5)
    tk.Label(popup, text="1. Nhập tên và chức vụ.\n"
                         "2. Nhìn vào camera.\n"
                         "3. Nhấn 'Chụp & Lưu' để enroll.\n"
                         "4. Nhấn 'Hủy' để thoát.", justify="left", bg="#f0f0f0").pack(pady=5)

    tk.Label(popup, text="Tên nhân viên:", font=("Arial",12)).pack(pady=5)
    name_var = tk.StringVar()
    tk.Entry(popup, textvariable=name_var, font=("Arial",12)).pack(pady=5)

    tk.Label(popup, text="Chức vụ:", font=("Arial",12)).pack(pady=5)
    position_var = tk.StringVar()
    tk.Entry(popup, textvariable=position_var, font=("Arial",12)).pack(pady=5)

    lbl_cam = tk.Label(popup)
    lbl_cam.pack(pady=10)

    cap = cv2.VideoCapture(0)

    def update_frame():
        ret, frame = cap.read()
        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image).resize((320,240))
            imgtk = ImageTk.PhotoImage(img)
            lbl_cam.imgtk = imgtk
            lbl_cam.config(image=imgtk)
            popup.after(30, update_frame)
        else:
            cap.release()
            lbl_cam.config(image="")

    def take_photo():
        name = name_var.get().strip()
        position = position_var.get().strip()
        if not name or not position:
            messagebox.showwarning("Lỗi","Vui lòng nhập tên và chức vụ")
            return
        ret, frame = cap.read()
        if not ret:
            messagebox.showwarning("Lỗi","Chưa có hình từ camera")
            return
        person_dir = os.path.join(DATASET_PATH, name)
        os.makedirs(person_dir, exist_ok=True)
        img_path = os.path.join(person_dir, f"{int(time.time())}.jpg")
        cv2.imwrite(img_path, frame)

        # Tính embedding trung bình
        imgs=[]
        for f in os.listdir(person_dir):
            try:
                img = Image.open(os.path.join(person_dir,f)).convert("RGB")
                imgs.append(get_embedding(img))
            except: continue
        if imgs:
            avg_emb = np.mean(np.array(imgs), axis=0)
            encodings, names, positions = load_encodings()
            encodings.append(avg_emb)
            names.append(name)
            positions.append(position)
            save_encodings(encodings, names, positions)
            messagebox.showinfo("Thành công", f"Đã enroll {name}")
            refresh_cards()
        cap.release()
        popup.destroy()

    def cancel():
        cap.release()
        popup.destroy()

    btn_frame = tk.Frame(popup)
    btn_frame.pack(pady=5)
    tk.Button(btn_frame, text="Chụp & Lưu", bg="#27ae60", fg="white", width=12, command=take_photo).pack(side="left", padx=10)
    tk.Button(btn_frame, text="Hủy", bg="#c0392b", fg="white", width=12, command=cancel).pack(side="left", padx=10)

    update_frame()  # Bắt đầu hiển thị live feed



# ---------- Recognize ----------
def recognize_callback(action="checkin"):
    global camera_running
    if camera_running:
        messagebox.showinfo("Thông báo","Camera đang chạy")
        return
    camera_running=True
    encodings,names,positions=load_encodings()
    if not encodings:
        messagebox.showwarning("Cảnh báo","Chưa có nhân viên enroll")
        camera_running=False
        return
    cap=cv2.VideoCapture(0)
    recognized_name=None
    messagebox.showinfo("Hướng dẫn","Nhìn vào camera. Nhận diện xong sẽ lưu thời gian")
    while True:
        ret,frame=cap.read()
        if not ret: continue
        img=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        face_emb=get_embedding(img)
        for known,name in zip(encodings,names):
            if compare_embeddings(face_emb,known):
                recognized_name=name
                break
        cv2.imshow("Nhận diện",frame)
        if recognized_name or cv2.waitKey(1)&0xFF==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    camera_running=False
    if recognized_name:
        attendance=load_attendance()
        now=datetime.now()
        records=attendance.get(recognized_name,[])
        if action=="checkin":
            records.append([now,None])
        else:
            if records and records[-1][1] is None:
                records[-1][1]=now
        attendance[recognized_name]=records
        save_attendance(attendance)
        messagebox.showinfo("Thành công",f"{action.title()} thành công cho {recognized_name}")
        refresh_cards()
    else:
        messagebox.showwarning("Thất bại","Không nhận diện được khuôn mặt")

# ---------- Sidebar Buttons ----------
create_sidebar_button("Enroll nhân viên mới","#2980b9",enroll_callback)
create_sidebar_button("Check-in","#27ae60",lambda:recognize_callback("checkin"))
create_sidebar_button("Check-out","#c0392b",lambda:recognize_callback("checkout"))
create_sidebar_button("Thoát","#7f8c8d",root.quit)

search_var.trace_add("write",refresh_cards)
refresh_cards()
root.mainloop()
