# Hướng dẫn Cài đặt Môi trường

## 1. Cài đặt Conda và Tạo Môi trường

```bash
# Tạo môi trường conda mới với Python 3.9
conda create -n face_recognition python=3.9
conda activate face_recognition
```

## 2. Cài đặt CUDA và cuDNN

```bash
# Cài đặt CUDA Toolkit 11.8
conda install -c conda-forge cudatoolkit=11.8

# Cài đặt cuDNN
conda install -c anaconda cudnn
```

## 3. Cài đặt các Package Python cần thiết

```bash
# Cài đặt PyTorch với CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Cài đặt numpy phiên bản cụ thể (quan trọng để tránh lỗi tương thích)
pip install numpy==1.24.3

# Cài đặt onnxruntime-gpu phiên bản cụ thể
pip install onnxruntime-gpu==1.13.1

# Cài đặt insightface
pip install insightface==0.7.3
```

## 4. Các Package Khác

```bash
pip install opencv-python
pip install tqdm
pip install scikit-learn
```

## 5. Cấu trúc Thư mục Dự án

```
project_root/
├── src/
│   ├── arc.py
│   ├── face_database.pkl
│   ├── frames.py
│   ├── realtime.py
│   ├── rotate_iphone_images.py
│   └── check_cuda.py
├── ATTENDANCE_DATASET/
│   ├── test/
│   ├── train/
│   └── val/
└── FRAME_DATASET/
    ├── test/
    ├── train/
    └── val/
```

## 6. Kiểm tra Cài đặt

Để kiểm tra xem môi trường đã được cài đặt đúng, chạy file `check_cuda.py`:

```python
# File: src/check_cuda.py
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import torch
import onnxruntime
import numpy as np

print("==== CUDA Availability Check ====")

# Kiểm tra CUDA với PyTorch
print("\n1. PyTorch CUDA Check:")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name()}")

# Kiểm tra ONNX Runtime
print("\n2. ONNX Runtime Providers:")
print(f"Available providers: {onnxruntime.get_available_providers()}")
print(f"ONNX Runtime version: {onnxruntime.__version__}")

# Kiểm tra cuDNN
if torch.cuda.is_available():
    print("\n3. cuDNN Check:")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
```

Chạy file kiểm tra:
```bash
python src/check_cuda.py
```

Kết quả mong đợi sẽ hiển thị:
- CUDA available: True
- CUDAExecutionProvider trong danh sách providers
- cuDNN enabled: True

## 7. Chạy Chương trình

Sau khi cài đặt xong, có thể chạy chương trình bằng lệnh:

```bash
# Chạy script chính
python src/arc.py

# Chạy ứng dụng real-time
python src/realtime.py

# Chạy evaluation chi tiết
python src/arc_with_evaluation.py
```

## Lưu ý Quan trọng

1. Phiên bản các package phải đúng như trong hướng dẫn để tránh lỗi tương thích:
   - numpy==1.24.3
   - onnxruntime-gpu==1.13.1
   - insightface==0.7.3

2. Nếu gặp lỗi OpenMP, đã được xử lý trong code bằng cách thêm:
   ```python
   os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
   ```

3. Cần có GPU NVIDIA với driver phiên bản tương thích CUDA 11.8

4. Đảm bảo đủ dung lượng ổ cứng cho dataset và model
