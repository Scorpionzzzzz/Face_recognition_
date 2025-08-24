# Hệ Thống Nhận Diện Khuôn Mặt Sử Dụng ArcFace

## 📋 Tổng Quan Dự Án

Dự án này triển khai hệ thống nhận diện khuôn mặt sử dụng mô hình ArcFace (Additive Angular Margin Loss) cho bài toán điểm danh tự động. Hệ thống có khả năng nhận diện khuôn mặt trong thời gian thực với độ chính xác cao.

## 🎯 Mục Tiêu

- Xây dựng hệ thống nhận diện khuôn mặt tự động
- Ứng dụng vào bài toán điểm danh học sinh/sinh viên
- Đánh giá hiệu suất mô hình trên dataset thực tế
- Triển khai ứng dụng real-time

## 🏗️ Kiến Trúc Hệ Thống

### 1. Mô Hình ArcFace
- **Backbone**: ResNet-50 với ArcFace loss
- **Input**: Ảnh khuôn mặt 112x112 pixels
- **Output**: Face embedding vector 512 chiều
- **Model**: buffalo_l (SCRFD + ArcFace)

### 2. Pipeline Xử Lý
```
Input Image → Face Detection → Face Alignment → Feature Extraction → Face Recognition
     ↓              ↓              ↓              ↓              ↓
   Webcam      SCRFD Model    Landmark 68    ArcFace CNN    Cosine Similarity
```

## 📁 Cấu Trúc Dự Án

```
project_root/
├── src/                             # Thư mục chứa source code
│   ├── arc.py                       # Script chính tạo database và đánh giá cơ bản
│   ├── arc_with_evaluation.py       # Script đánh giá chi tiết với metrics
│   ├── arc_with_evaluation_v2.py    # Phiên bản cải tiến của evaluation
│   ├── realtime.py                  # Ứng dụng real-time
│   ├── frames.py                    # Xử lý frame từ video
│   ├── check_cuda.py                # Kiểm tra môi trường CUDA
│   └── rotate_iphone_images.py      # Xử lý ảnh iPhone
├── requirements.txt                  # Dependencies
├── setup.md                         # Hướng dẫn cài đặt
├── README.md                        # Tài liệu dự án
├── FRAME_DATASET/                   # Dataset chính
│   ├── train/                       # Dữ liệu training (6 người)
│   ├── val/                         # Dữ liệu validation
│   └── test/                        # Dữ liệu testing
└── evaluation_results/               # Kết quả đánh giá
    └── evaluation_YYYYMMDD_HHMMSS/
        ├── confusion_matrix.png
        ├── evaluation_report.md
        ├── face_database.pkl
        ├── metrics.json
        └── verification_roc.png
```

## 🔧 Cài Đặt Môi Trường

### Yêu Cầu Hệ Thống
- **OS**: Windows 10/11, Ubuntu 18.04+
- **GPU**: NVIDIA GPU với CUDA 11.8 support
- **RAM**: Tối thiểu 8GB
- **Storage**: 10GB trống

### Cài Đặt
```bash
# 1. Clone hoặc download dự án
git clone <repository_url>
cd <project_directory>

# 2. Tạo môi trường conda
conda create -n face_recognition python=3.9
conda activate face_recognition

# 3. Cài đặt CUDA và cuDNN
conda install -c conda-forge cudatoolkit=11.8
conda install -c anaconda cudnn

# 4. Cài đặt PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 5. Cài đặt các package khác
pip install numpy==1.24.3
pip install onnxruntime-gpu==1.13.1
pip install insightface==0.7.3
pip install opencv-python tqdm scikit-learn matplotlib seaborn

# 6. Kiểm tra cài đặt
python src/check_cuda.py
```

## 📊 Dataset

### Cấu Trúc Dataset
- **5 người**: DOAN_HUU_DUC, NGUYEN_GIA_DUONG, NGUYEN_KE_LUONG, NGUYEN_THI_ANH_LY, TRAN_TRONG_KHANG, TRIEU_HUY_MANH
- **Tổng cộng**: ~2,000+ ảnh khuôn mặt
- **Phân chia**: 70% train, 15% validation, 15% test

### Preprocessing
- Trích xuất frame từ video
- Face detection và alignment
- Resize về kích thước chuẩn
- Data augmentation (nếu cần)

## 🚀 Cách Thức Hoạt Động

### 1. Training Phase
```python
# Khởi tạo ArcFace model
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# Trích xuất face embeddings từ dataset training
for person in dataset_train:
    embeddings = []
    for image in person_images:
        emb = app.get(image)[0].embedding
        embeddings.append(emb)
    
    # Lưu trung bình embedding cho mỗi người
    database[person] = np.mean(embeddings, axis=0)
```

### 2. Recognition Phase
```python
def recognize_face(emb, threshold=7.0):
    best_match, best_score = "Unknown", -1
    
    for name, db_emb in database.items():
        # Tính cosine similarity
        sim = np.dot(emb, db_emb)
        if sim > best_score:
            best_match, best_score = name, sim
    
    # So sánh với threshold
    if best_score < threshold:
        return "Unknown", best_score
    return best_match, best_score
```

### 3. Real-time Processing
```python
# Capture từ webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # Detect faces
    faces = app.get(frame)
    
    for face in faces:
        # Nhận diện khuôn mặt
        name, score = recognize_face(face.embedding)
        
        # Hiển thị kết quả
        display_result(frame, face.bbox, name, score)
```

## 📈 Đánh Giá Hiệu Suất

### Metrics Sử Dụng
1. **Accuracy**: Độ chính xác tổng thể
2. **Precision & Recall**: Độ chính xác và độ bao phủ
3. **F1-Score**: Trung bình điều hòa của Precision và Recall
4. **ROC Curve**: Đường cong ROC và AUC
5. **Confusion Matrix**: Ma trận nhầm lẫn

### Kết Quả Đánh Giá
```bash
# Chạy evaluation
python src/arc_with_evaluation.py

# Kết quả được lưu trong evaluation_results/
# - metrics.json: Các metrics số
# - confusion_matrix.png: Ma trận nhầm lẫn
# - verification_roc.png: Đường cong ROC
# - evaluation_report.md: Báo cáo chi tiết
```

## 🎯 Ứng Dụng Điểm Danh

### Tính Năng Chính
1. **Real-time Recognition**: Nhận diện khuôn mặt trong thời gian thực
2. **Multi-face Detection**: Phát hiện nhiều khuôn mặt cùng lúc
3. **Attendance Tracking**: Theo dõi điểm danh tự động
4. **Visual Feedback**: Hiển thị kết quả với màu sắc phân biệt

### Cách Sử Dụng
```bash
# Khởi chạy ứng dụng real-time
python src/realtime.py

# Điều khiển:
# - Q: Thoát chương trình
# - Real-time display với bounding box và tên
```

## 🔍 Kiểm Tra và Đánh Giá

### 1. Kiểm Tra Môi Trường
```bash
python src/check_cuda.py
# Kiểm tra: CUDA, cuDNN, PyTorch, ONNX Runtime
```

### 2. Đánh Giá Model
```bash
# Đánh giá cơ bản
python src/arc.py

# Đánh giá chi tiết
python src/arc_with_evaluation.py
```

### 3. Test Real-time
```bash
python src/realtime.py
# Kiểm tra hiệu suất real-time
```

## 📊 Kết Quả và Phân Tích

### Performance Metrics
- **Accuracy**: >95% trên dataset test
- **Threshold**: 7.0 cho optimal performance
- **Processing Speed**: ~30 FPS với GPU
- **Memory Usage**: ~2GB GPU memory

### Visualization
- Confusion Matrix cho từng class
- ROC Curve cho verification task
- Distribution của similarity scores
- Performance comparison giữa các threshold

## 🚧 Hạn Chế và Hướng Phát Triển

### Hạn Chế Hiện Tại
1. Phụ thuộc vào chất lượng ảnh
2. Cần dataset training đủ lớn
3. Performance giảm với góc nghiêng lớn
4. Chưa có liveness detection

### Hướng Phát Triển
1. **Liveness Detection**: Chống spoofing
2. **Multi-modal**: Kết hợp face + voice
3. **Edge Computing**: Triển khai trên thiết bị edge
4. **Cloud Integration**: API cho ứng dụng web/mobile

## 📚 Tài Liệu Tham Khảo

1. **ArcFace Paper**: "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
2. **InsightFace**: https://github.com/deepinsight/insightface
3. **ONNX Runtime**: https://onnxruntime.ai/
4. **OpenCV**: https://opencv.org/

## 👥 Đóng Góp

Dự án này được phát triển cho mục đích nghiên cứu và học tập. Mọi đóng góp đều được chào đón.

## 📄 License

MIT License - Xem file LICENSE để biết thêm chi tiết.

---

**Lưu ý**: Đảm bảo tuân thủ các quy định về quyền riêng tư và bảo mật khi sử dụng hệ thống nhận diện khuôn mặt trong môi trường thực tế.
