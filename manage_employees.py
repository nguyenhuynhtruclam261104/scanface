import os
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
from datetime import datetime
import numpy as np
import face_recognition
import threading
import pickle

# --------- Paths ---------
DATASET_PATH = "dataset_dl"
ENCODINGS_PATH = "encodings_dl.pkl"
ATTENDANCE_PATH = "attendance_dl.pkl"
os.makedirs(DATASET_PATH, exist_ok=True)

# --------- Utils ---------
def load_encodings():
    if not os.path.exists(ENCODINGS_PATH):
        return [], [], []
    with open(ENCODINGS_PATH, "rb") as f:
        data = pickle.load(f)
    return data.get("encodings",[]), data.get("names",[]), data.get("positions",[])

def save_encodings(path, encodings, names, positions):
    data = {"encodings":encodings, "names":names, "positions":positions}
    with open(path, "wb") as f:
        pickle.dump(data,f)

def load_attendance():
    if not os.path.exists(ATTENDANCE_PATH):
        return {}
    with open(ATTENDANCE_PATH,"rb") as f:
        return pickle.load(f)

def save_attendance(att):
    with open(ATTENDANCE_PATH,"wb") as f:
        pickle.dump(att,f)

# --------- DL Embedding ---------
def get_embedding(pil_image):
    image_np = np.array(pil_image)
    face_locations = face_recognition.face_locations(image_np)
    if not face_locations:
        return None
    face_encodings = face_recognition.face_encodings(image_np, known_face_locations=face_locations)
    return face_encodings[0]

def compare_embeddings(emb1, emb2, threshold=0.6):
    return np.linalg.norm(np.array(emb1)-np.array(emb2)) < threshold

# --------- GUI ---------
root = tk.Tk()
root.title("Face Recognition Attendance DL")
root.geometry("1250x650")

top_frame = tk.Frame(root)
top_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

search_var = tk.StringVar()
tk.Label(top_frame, text="Tìm kiếm:").pack(side=tk.LEFT,padx=5)
search_entry = tk.Entry(top_frame, textvariable=search_var)
search_entry.pack(side=tk.LEFT,padx=5)

columns = ("Tên","Chức vụ","Tổng giờ","10 lần gần nhất")
tree = ttk.Treeview(root, columns=columns, show="headings", height=20)
for col in columns:
    tree.heading(col, text=col)
    tree.column(col,width=280)
tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

def refresh_tree(*args):
    tree.delete(*tree.get_children())
    encodings, names, positions_list = load_encodings()
    attendance = load_attendance()
    filter_text = search_var.get().lower()
    unique_emp = {}
    for n,pos in zip(names, positions_list):
        if n not in unique_emp:
            unique_emp[n] = pos
    for name, pos in unique_emp.items():
        if filter_text in name.lower():
            total_hours = 0.0
            records = attendance.get(name, [])
            for rec in records:
                in_time, out_time = rec
                if out_time:
                    total_hours += (out_time - in_time).total_seconds()/3600
            last10 = records[-10:] if records else []
            last10_str = "\n".join([f"{r[0].strftime('%d/%m %H:%M')} - {r[1].strftime('%d/%m %H:%M') if r[1] else '...'}" for r in last10])
            tree.insert("", tk.END, values=(name,pos,f"{total_hours:.2f}h", last10_str))

search_var.trace_add("write", refresh_tree)

# --------- Enroll Employee ---------
def enroll_callback():
    popup = tk.Toplevel(root)
    popup.title("Enroll Employee")
    popup.geometry("350x180")
    
    tk.Label(popup, text="Tên nhân viên:").pack(pady=5)
    name_var = tk.StringVar()
    tk.Entry(popup, textvariable=name_var).pack()
    
    tk.Label(popup, text="Chức vụ:").pack(pady=5)
    position_var = tk.StringVar()
    tk.Entry(popup, textvariable=position_var).pack()
    
    def on_ok():
        name = name_var.get().strip()
        position = position_var.get().strip()
        if not name or not position:
            messagebox.showwarning("Lỗi","Vui lòng nhập đầy đủ tên và chức vụ")
            return
        popup.destroy()
        threading.Thread(target=capture_and_save, args=(name,position)).start()
    
    tk.Button(popup, text="OK", command=on_ok).pack(pady=10)

def capture_and_save(name, position, num_images=30):
    person_dir = os.path.join(DATASET_PATH,name)
    os.makedirs(person_dir,exist_ok=True)
    cap = cv2.VideoCapture(0)
    count = 0
    messagebox.showinfo("Hướng dẫn", f"Chụp {num_images} ảnh cho nhân viên {name}\nNhấn 'c' để chụp, 'q' để thoát")
    while count<num_images:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Chụp ảnh - Nhấn 'c' để chụp", frame)
        key = cv2.waitKey(1)
        if key==ord('c'):
            img_path = os.path.join(person_dir,f"img{count+1}.jpg")
            cv2.imwrite(img_path, frame)
            count += 1
        elif key==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
    encodings, names, positions_list = load_encodings()
    for img_file in os.listdir(person_dir):
        img = Image.open(os.path.join(person_dir,img_file)).convert('RGB')
        emb = get_embedding(img)
        if emb is not None:
            encodings.append(emb)
            names.append(name)
            positions_list.append(position)
    save_encodings(ENCODINGS_PATH, encodings, names, positions_list)
    messagebox.showinfo("Enroll", f"Nhân viên {name} ({position}) đã enroll thành công!")
    refresh_tree()

# --------- Check-in / Check-out bằng tên ---------
def check_in_callback():
    name = simpledialog.askstring("Check-in","Nhập tên nhân viên (demo)")
    if not name: return
    attendance = load_attendance()
    now = datetime.now()
    records = attendance.get(name,[])
    records.append([now,None])
    attendance[name]=records
    save_attendance(attendance)
    messagebox.showinfo("Check-in", f"{name} đã check-in lúc {now.strftime('%H:%M:%S')}")
    refresh_tree()

def check_out_callback():
    name = simpledialog.askstring("Check-out","Nhập tên nhân viên (demo)")
    if not name: return
    attendance = load_attendance()
    now = datetime.now()
    records = attendance.get(name,[])
    if records and records[-1][1] is None:
        records[-1][1] = now
        attendance[name]=records
        save_attendance(attendance)
        messagebox.showinfo("Check-out", f"{name} đã check-out lúc {now.strftime('%H:%M:%S')}")
    else:
        messagebox.showwarning("Check-out","Không tìm thấy bản check-in chưa check-out")
    refresh_tree()

# --------- Check-in / Check-out bằng khuôn mặt ---------
def recognize_face_callback(is_checkin=True):
    encodings, names, positions_list = load_encodings()
    if not encodings:
        messagebox.showwarning("Lỗi","Chưa có nhân viên nào enroll")
        return
    cap = cv2.VideoCapture(0)
    recognized_name = None
    messagebox.showinfo("Hướng dẫn","Nhìn vào camera để nhận diện khuôn mặt.\nNhấn 'q' để thoát")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        rgb_frame = frame[:,:,::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        for face_emb in face_encodings:
            for known_emb,name in zip(encodings,names):
                if compare_embeddings(face_emb,known_emb):
                    recognized_name=name
                    break
        cv2.imshow("Nhận diện", frame)
        key=cv2.waitKey(1)
        if key==ord('q') or recognized_name:
            break
    cap.release()
    cv2.destroyAllWindows()
    if recognized_name:
        if is_checkin:
            attendance = load_attendance()
            now=datetime.now()
            records=attendance.get(recognized_name,[])
            records.append([now,None])
            attendance[recognized_name]=records
            save_attendance(attendance)
            messagebox.showinfo("Check-in", f"{recognized_name} đã check-in lúc {now.strftime('%H:%M:%S')}")
        else:
            attendance = load_attendance()
            now=datetime.now()
            records=attendance.get(recognized_name,[])
            if records and records[-1][1] is None:
                records[-1][1]=now
                attendance[recognized_name]=records
                save_attendance(attendance)
                messagebox.showinfo("Check-out", f"{recognized_name} đã check-out lúc {now.strftime('%H:%M:%S')}")
            else:
                messagebox.showwarning("Check-out","Không tìm thấy bản check-in chưa check-out")
        refresh_tree()
    else:
        messagebox.showwarning("Nhận diện","Không nhận diện được khuôn mặt")

# --------- Buttons ---------
tk.Button(top_frame,text="Enroll Employee",width=15,command=enroll_callback).pack(side=tk.LEFT,padx=5)
tk.Button(top_frame,text="Check-in (Tên)",width=15,command=check_in_callback).pack(side=tk.LEFT,padx=5)
tk.Button(top_frame,text="Check-out (Tên)",width=15,command=check_out_callback).pack(side=tk.LEFT,padx=5)
tk.Button(top_frame,text="Check-in (Khuôn mặt)",width=20,command=lambda: recognize_face_callback(True)).pack(side=tk.LEFT,padx=5)
tk.Button(top_frame,text="Check-out (Khuôn mặt)",width=20,command=lambda: recognize_face_callback(False)).pack(side=tk.LEFT,padx=5)
tk.Button(top_frame,text="Thoát",width=10,command=root.destroy).pack(side=tk.RIGHT,padx=5)

# --------- Double click xem ảnh nhân viên ---------
def on_tree_double_click(event):
    item = tree.selection()
    if not item: return
    values = tree.item(item,"values")
    name = values[0]
    person_dir = os.path.join(DATASET_PATH,name)
    if not os.path.exists(person_dir):
        messagebox.showwarning("Lỗi",f"Không tìm thấy thư mục ảnh của {name}")
        return
    files=os.listdir(person_dir)
    if not files:
        messagebox.showwarning("Lỗi",f"Không có ảnh nào của {name}")
        return
    popup=tk.Toplevel(root)
    popup.title(f"Ảnh nhân viên {name}")
    canvas=tk.Canvas(popup,width=800,height=600,scrollregion=(0,0,1600,1200))
    hbar=tk.Scrollbar(popup,orient=tk.HORIZONTAL,command=canvas.xview)
    hbar.pack(side=tk.BOTTOM,fill=tk.X)
    vbar=tk.Scrollbar(popup,orient=tk.VERTICAL,command=canvas.yview)
    vbar.pack(side=tk.RIGHT,fill=tk.Y)
    canvas.config(xscrollcommand=hbar.set,yscrollcommand=vbar.set)
    canvas.pack(side=tk.LEFT,expand=True,fill=tk.BOTH)
    x,y=10,10
    for f in files:
        img = Image.open(os.path.join(person_dir,f)).resize((200,200))
        photo = ImageTk.PhotoImage(img)
        canvas.create_image(x,y,anchor=tk.NW,image=photo)
        canvas.image=photo
        x+=210
        if x>1500:
            x=10
            y+=210
    attendance = load_attendance()
    records = attendance.get(name,[])
    last10 = records[-10:] if records else []
    text = "\n".join([f"{r[0].strftime('%d/%m %H:%M')} - {r[1].strftime('%d/%m %H:%M') if r[1] else '...'}" for r in last10])
    tk.Label(popup,text=f"10 lần check gần nhất:\n{text}").pack(side=tk.BOTTOM,pady=5)

tree.bind("<Double-1>",on_tree_double_click)

refresh_tree()
root.mainloop()
