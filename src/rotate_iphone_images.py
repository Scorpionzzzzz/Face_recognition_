import os
from PIL import Image

# Danh sách tên người cần xoay ảnh
TARGETS = ['TRAN_TRONG_KHANG', 'DOAN_HUU_DUC']
# Các thư mục cần xử lý
directories = [
    '../FRAME_DATASET/train',
    '../FRAME_DATASET/test',
    '../FRAME_DATASET/val',
]

# Hàm xoay ảnh 90 độ ngược chiều kim đồng hồ

def rotate_image_90ccw(image_path):
    with Image.open(image_path) as img:
        return img.rotate(90, expand=True)


def process_person_images(person_dir):
    for root, _, files in os.walk(person_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, file)
                try:
                    rotated_img = rotate_image_90ccw(img_path)
                    rotated_img.save(img_path)
                    print(f"Rotated: {img_path}")
                except Exception as e:
                    print(f"Error rotating {img_path}: {e}")


def main():
    for dataset_dir in directories:
        for person in TARGETS:
            person_dir = os.path.join(dataset_dir, person)
            if os.path.exists(person_dir):
                process_person_images(person_dir)
            else:
                print(f"Not found: {person_dir}")

if __name__ == "__main__":
    main()
