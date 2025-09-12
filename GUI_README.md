# 🎯 Hướng Dẫn Sử Dụng Giao Diện Điểm Danh PyQt5

## 📋 Tổng Quan

Giao diện điểm danh sử dụng PyQt5 được thiết kế để cung cấp trải nghiệm người dùng chuyên nghiệp cho hệ thống nhận diện khuôn mặt ArcFace. Giao diện bao gồm 3 tab chính:

1. **🎥 Real-time Recognition**: Nhận diện khuôn mặt thời gian thực
2. **📊 Lịch Sử Điểm Danh**: Quản lý và xem lịch sử điểm danh
3. **⚙️ Quản Lý**: Quản lý hệ thống và database

## 🚀 Cài Đặt và Chạy

### 1. Cài Đặt Dependencies

```bash
# Cài đặt PyQt5 và các package cần thiết
pip install PyQt5 pandas openpyxl

# Hoặc cài đặt từ requirements.txt
pip install -r requirements.txt
```

### 2. Chạy Giao Diện

#### Chạy Giao Diện Đầy Đủ (với ArcFace)
```bash
python src/attendance_gui.py
```

#### Chạy Giao Diện Demo (không cần ArcFace)
```bash
python src/demo_gui.py
```

## 🎥 Tab Real-time Recognition

### Tính Năng Chính
- **Camera Feed**: Hiển thị video stream từ webcam
- **Face Detection**: Phát hiện và nhận diện khuôn mặt tự động
- **Recognition Info**: Hiển thị tên và điểm số nhận diện
- **Quick Actions**: Điểm danh nhanh và xóa kết quả

### Cách Sử Dụng
1. **Bắt Đầu**: Nhấn nút "▶️ Bắt Đầu" để khởi động camera
2. **Nhận Diện**: Hệ thống sẽ tự động phát hiện và nhận diện khuôn mặt
3. **Điểm Danh**: Nhấn "✅ Điểm Danh" khi nhận diện thành công
4. **Dừng**: Nhấn "⏹️ Dừng" để tắt camera

### Cài Đặt
- **Threshold**: Điều chỉnh ngưỡng nhận diện (mặc định: 7.0)
- **Camera Resolution**: 640x480 pixels

## 📊 Tab Lịch Sử Điểm Danh

### Tính Năng Chính
- **Attendance Table**: Bảng hiển thị lịch sử điểm danh
- **Filter Controls**: Lọc theo ngày và người
- **Export Excel**: Xuất dữ liệu ra file Excel
- **Clear History**: Xóa lịch sử điểm danh

### Cấu Trúc Bảng
| Cột | Mô Tả |
|-----|--------|
| **Thời Gian** | Ngày và giờ điểm danh |
| **Tên** | Tên người được điểm danh |
| **Trạng Thái** | Trạng thái điểm danh (Present) |
| **Điểm Số** | Độ tin cậy nhận diện |
| **Hành Động** | Loại điểm danh (Auto/Demo) |

### Cách Sử Dụng
1. **Lọc Dữ Liệu**: Chọn ngày và người cần lọc
2. **Áp Dụng Bộ Lọc**: Nhấn "🔍 Áp Dụng Bộ Lọc"
3. **Xuất Excel**: Nhấn "📤 Xuất Excel" để lưu dữ liệu
4. **Xóa Lịch Sử**: Nhấn "🗑️ Xóa Lịch Sử" (có xác nhận)

## ⚙️ Tab Quản Lý

### Tính Năng Chính
- **Database Info**: Thông tin về face database
- **System Status**: Trạng thái các thành phần hệ thống
- **Management Actions**: Các thao tác quản lý

### Thông Tin Database
- **Số người**: Tổng số người trong database
- **Danh sách**: Tên các người được đăng ký

### Trạng Thái Hệ Thống
- **ArcFace Model**: Trạng thái mô hình nhận diện
- **Camera**: Trạng thái camera
- **Recognition Thread**: Trạng thái thread nhận diện

### Thao Tác Quản Lý
1. **Tải Lại Database**: Nhấn "🔄 Tải Lại Database"
2. **Kiểm Tra Camera**: Nhấn "📷 Kiểm Tra Camera"

## 🎨 Giao Diện và Thiết Kế

### Màu Sắc
- **Primary**: #4a90e2 (Xanh dương)
- **Success**: #28a745 (Xanh lá)
- **Warning**: #ffc107 (Vàng)
- **Danger**: #dc3545 (Đỏ)
- **Background**: #f0f0f0 (Xám nhạt)

### Layout
- **Responsive Design**: Giao diện thích ứng với kích thước màn hình
- **Tab-based Navigation**: Chuyển đổi giữa các chức năng dễ dàng
- **Status Bar**: Hiển thị thông tin trạng thái và thông báo

### Icons và Emojis
- Sử dụng emojis để tăng tính trực quan
- Icons rõ ràng cho các chức năng chính
- Màu sắc phân biệt cho các trạng thái khác nhau

## 🔧 Cấu Hình và Tùy Chỉnh

### Threshold Settings
```python
# Trong attendance_gui.py
self.threshold_spin.setRange(1, 20)  # Phạm vi threshold
self.threshold_spin.setValue(7)       # Giá trị mặc định
```

### Camera Settings
```python
# Độ phân giải camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

### Database Path
```python
# Đường dẫn database mặc định
eval_dirs = glob.glob("evaluation_results/evaluation_*")
latest_eval_dir = max(eval_dirs)
database_path = os.path.join(latest_eval_dir, "face_database.pkl")
```

## 📱 Tính Năng Nâng Cao

### Multi-threading
- **Face Recognition Thread**: Chạy riêng biệt để không block GUI
- **Signal-based Communication**: Sử dụng PyQt signals để giao tiếp giữa threads
- **Thread Safety**: Đảm bảo an toàn khi truy cập dữ liệu chung

### Data Persistence
- **JSON Storage**: Lưu attendance data dưới dạng JSON
- **Auto-save**: Tự động lưu khi đóng ứng dụng
- **Backup Support**: Hỗ trợ backup và restore dữ liệu

### Error Handling
- **Exception Handling**: Xử lý lỗi gracefully
- **User Feedback**: Thông báo lỗi rõ ràng cho người dùng
- **Fallback Mechanisms**: Cơ chế dự phòng khi có lỗi

## 🚨 Xử Lý Sự Cố

### Lỗi Thường Gặp

#### 1. Camera Không Hoạt Động
```bash
# Kiểm tra camera
python src/attendance_gui.py
# Vào tab Quản Lý → Kiểm Tra Camera
```

#### 2. Database Không Tải Được
```bash
# Kiểm tra thư mục evaluation_results
ls evaluation_results/
# Đảm bảo có file face_database.pkl
```

#### 3. ArcFace Model Lỗi
```bash
# Kiểm tra CUDA và dependencies
python src/check_cuda.py
# Cài đặt lại insightface nếu cần
pip install insightface==0.7.3
```

### Debug Mode
```python
# Thêm logging chi tiết
import logging
logging.basicConfig(level=logging.DEBUG)

# Hoặc sử dụng print statements
print(f"Debug: {variable_name}")
```

## 📈 Performance và Tối Ưu

### Memory Management
- **Efficient Image Processing**: Xử lý ảnh tối ưu với OpenCV
- **Garbage Collection**: Tự động dọn dẹp bộ nhớ
- **Resource Cleanup**: Giải phóng tài nguyên khi đóng ứng dụng

### CPU/GPU Optimization
- **GPU Acceleration**: Sử dụng CUDA cho ArcFace
- **Thread Management**: Quản lý threads hiệu quả
- **Frame Rate Control**: Kiểm soát FPS để tối ưu performance

## 🔒 Bảo Mật và Quyền Riêng Tư

### Data Protection
- **Local Storage**: Dữ liệu được lưu trữ locally
- **No Cloud Sync**: Không đồng bộ lên cloud
- **User Consent**: Yêu cầu sự đồng ý trước khi sử dụng

### Access Control
- **Admin Mode**: Chế độ quản trị viên cho các thao tác nhạy cảm
- **User Permissions**: Phân quyền người dùng
- **Audit Log**: Ghi log các thao tác quan trọng

## 🚀 Hướng Phát Triển

### Tính Năng Tương Lai
1. **Multi-language Support**: Hỗ trợ đa ngôn ngữ
2. **Cloud Integration**: Đồng bộ dữ liệu lên cloud
3. **Mobile App**: Ứng dụng mobile companion
4. **API Integration**: REST API cho tích hợp hệ thống

### Scalability
- **Database Optimization**: Tối ưu hóa database cho dataset lớn
- **Load Balancing**: Cân bằng tải cho nhiều camera
- **Microservices**: Kiến trúc microservices

## 📞 Hỗ Trợ và Liên Hệ

### Documentation
- **Code Comments**: Code được comment chi tiết
- **API Reference**: Tài liệu API đầy đủ
- **Examples**: Ví dụ sử dụng cụ thể

### Community
- **GitHub Issues**: Báo cáo lỗi và feature requests
- **Discussion Forum**: Diễn đàn thảo luận
- **Contributing Guide**: Hướng dẫn đóng góp

---

**Lưu ý**: Giao diện này được thiết kế để hoạt động với hệ thống ArcFace hiện có. Đảm bảo rằng tất cả dependencies và models đã được cài đặt đúng cách trước khi sử dụng.
