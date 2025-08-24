import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import cv2
import numpy as np
from insightface.app import FaceAnalysis

# =====================
# 1. Khởi tạo ArcFace
# =====================
app = FaceAnalysis(name="buffalo_l")   # model buffalo_l (SCRFD + ArcFace)
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 GPU, -1 CPU

# =====================
# 2. Load database embeddings từ thư mục evaluation mới nhất
# =====================
import pickle
import glob

# Tìm thư mục evaluation mới nhất (đi ra ngoài thư mục src/)
eval_dirs = glob.glob("../evaluation_results/evaluation_*")
if not eval_dirs:
    raise FileNotFoundError("Không tìm thấy thư mục evaluation_results!")

latest_eval_dir = max(eval_dirs)  # Lấy thư mục có timestamp mới nhất
database_path = os.path.join(latest_eval_dir, "face_database.pkl")

if not os.path.exists(database_path):
    raise FileNotFoundError(f"Không tìm thấy file database trong {latest_eval_dir}")

with open(database_path, "rb") as f:
    database = pickle.load(f)

print(f"✅ Loaded database from: {database_path}")
print(f"✅ Found {len(database)} people in database: {list(database.keys())}")

print("✅ Loaded database for:", list(database.keys()))

# =====================
# 3. Hàm so sánh với threshold 10
# =====================
def recognize_face(emb, threshold=7.0):
    best_match, best_score = "Unknown", -1
    for name, db_emb in database.items():
        # Tính similarity score (sẽ cho số cao hơn)
        sim = np.dot(emb, db_emb)  
        if sim > best_score:
            best_match, best_score = name, sim
    if best_score < threshold:
        return "Unknown", best_score
    return best_match, best_score

# =====================
# 4. Webcam realtime
# =====================
cap = cv2.VideoCapture(0)  # 0 = webcam mặc định

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # detect & embed
    faces = app.get(frame)

    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int)
        name, score = recognize_face(f.embedding)

        # Chọn màu dựa trên kết quả nhận diện
        if name == "Unknown":
            color = (0, 0, 255)  # Màu đỏ cho Unknown
        else:
            color = (0, 255, 0)  # Màu xanh cho người đã biết

        # Vẽ bounding box + tên
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{name} ({score:.2f})",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Recognition Realtime", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
