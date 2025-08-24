import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import cv2
import numpy as np
import pickle
from tqdm import tqdm
from insightface.app import FaceAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import json

# Tạo thư mục cho kết quả đánh giá
EVAL_DIR = "evaluation_results"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
CURRENT_EVAL_DIR = os.path.join(EVAL_DIR, f"evaluation_{timestamp}")
os.makedirs(CURRENT_EVAL_DIR, exist_ok=True)

# =====================
# 1. Khởi tạo ArcFace
# =====================
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640,640))

def get_embedding(img):
    faces = app.get(img)
    if len(faces) == 0:
        return None
    return faces[0].embedding

# =====================
# 2. Tạo và lưu database
# =====================
DATASET_TRAIN = "../FRAME_DATASET/train"
database = {}
embeddings_per_person = {}  # Lưu tất cả embedding cho mỗi người

for person in os.listdir(DATASET_TRAIN):
    folder = os.path.join(DATASET_TRAIN, person)
    if not os.path.isdir(folder):
        continue

    embeddings = []
    for img_name in tqdm(os.listdir(folder), desc=f"Building {person}"):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        emb = get_embedding(img)
        if emb is not None:
            embeddings.append(emb)

    if len(embeddings) > 0:
        database[person] = np.mean(embeddings, axis=0)
        embeddings_per_person[person] = embeddings

print("✅ Đã tạo database cho:", list(database.keys()))

# Lưu database
with open("../face_database.pkl", "wb") as f:
    pickle.dump(database, f)

# =====================
# 3. Hàm đánh giá khoảng cách
# =====================
def calculate_distances(emb, database):
    distances = {}
    for name, db_emb in database.items():
        sim = np.dot(emb, db_emb)
        distances[name] = sim
    return distances

# =====================
# 4. Đánh giá trên tập test
# =====================
DATASET_TEST = "../FRAME_DATASET/test"

y_true = []
y_pred = []
y_scores = []  # Điểm similarity cho ROC curve
all_distances = []  # Lưu khoảng cách cho phân tích phân phối

for person in os.listdir(DATASET_TEST):
    folder = os.path.join(DATASET_TEST, person)
    if not os.path.isdir(folder):
        continue

    for img_name in tqdm(os.listdir(folder), desc=f"Testing {person}"):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        emb = get_embedding(img)
        if emb is None:
            continue

        # Tính toán khoảng cách với tất cả người trong database
        distances = calculate_distances(emb, database)
        pred_name = max(distances.items(), key=lambda x: x[1])[0]
        pred_score = distances[pred_name]

        y_true.append(person)
        y_pred.append(pred_name)
        y_scores.append(pred_score)
        all_distances.append(distances)

# =====================
# 5. Tạo báo cáo đánh giá
# =====================

# 5.1 Độ chính xác tổng thể
accuracy = accuracy_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred, labels=list(database.keys()))
class_report = classification_report(y_true, y_pred, labels=list(database.keys()), output_dict=True)

# 5.2 Vẽ Confusion Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=database.keys(),
            yticklabels=database.keys())
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(CURRENT_EVAL_DIR, 'confusion_matrix.png'))
plt.close()

# 5.3 Vẽ phân phối điểm similarity
plt.figure(figsize=(10, 6))
correct_scores = []
incorrect_scores = []
for true, pred, score in zip(y_true, y_pred, y_scores):
    if true == pred:
        correct_scores.append(float(score))
    else:
        incorrect_scores.append(float(score))

if len(correct_scores) > 0:
    plt.hist(correct_scores, bins=min(50, len(correct_scores)), 
             alpha=0.5, label='Correct Predictions', density=True)
if len(incorrect_scores) > 0:
    plt.hist(incorrect_scores, bins=min(50, len(incorrect_scores)), 
             alpha=0.5, label='Incorrect Predictions', density=True)

plt.xlabel('Similarity Score')
plt.ylabel('Density')
plt.title('Distribution of Similarity Scores')
plt.legend()
plt.savefig(os.path.join(CURRENT_EVAL_DIR, 'similarity_distribution.png'))
plt.close()

# 5.4 Tính ROC curve cho mỗi người
def calculate_roc_for_person(person):
    binary_truth = [1 if t == person else 0 for t in y_true]
    person_scores = []
    for distances in all_distances:
        person_scores.append(distances.get(person, 0))
    
    # Kiểm tra xem có đủ dữ liệu không
    if len(set(binary_truth)) < 2:  # Cần cả positive và negative samples
        print(f"WARNING: Không đủ dữ liệu để tính ROC curve cho {person}")
        return None, None, 0
        
    try:
        fpr, tpr, _ = roc_curve(binary_truth, person_scores)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc
    except Exception as e:
        print(f"WARNING: Không thể tính ROC curve cho {person}: {str(e)}")
        return None, None, 0

# Vẽ ROC curves
plt.figure(figsize=(10, 10))
valid_curves = False

for person in database.keys():
    fpr, tpr, roc_auc = calculate_roc_for_person(person)
    if fpr is not None and tpr is not None:
        plt.plot(fpr, tpr, label=f'{person} (AUC = {roc_auc:.2f})')
        valid_curves = True

if valid_curves:
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Person')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(CURRENT_EVAL_DIR, 'roc_curves.png'))
else:
    print("WARNING: Không có đủ dữ liệu để vẽ ROC curves")
plt.close()

# 5.5 Tính toán và lưu các metric
def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def safe_mean(arr):
    if not arr:
        return 0.0
    return float(np.mean(arr))

def safe_std(arr):
    if len(arr) < 2:  # Cần ít nhất 2 phần tử để tính std
        return 0.0
    return float(np.std(arr))

# Xử lý trường hợp không có dự đoán đúng hoặc sai
avg_correct = safe_mean(correct_scores)
avg_incorrect = safe_mean(incorrect_scores)
std_correct = safe_std(correct_scores)
std_incorrect = safe_std(incorrect_scores)

print(f"\nThống kê cơ bản:")
print(f"Số lượng dự đoán đúng: {len(correct_scores)}")
print(f"Số lượng dự đoán sai: {len(incorrect_scores)}")
print(f"Độ tương đồng trung bình (dự đoán đúng): {avg_correct:.4f} ± {std_correct:.4f}")
print(f"Độ tương đồng trung bình (dự đoán sai): {avg_incorrect:.4f} ± {std_incorrect:.4f}")

metrics = {
    'accuracy': float(accuracy),
    'classification_report': {k: {mk: float(mv) for mk, mv in v.items()} 
                            if isinstance(v, dict) else v 
                            for k, v in class_report.items()},
    'average_similarity_correct': float(avg_correct),
    'average_similarity_incorrect': float(avg_incorrect),
    'std_similarity_correct': float(std_correct),
    'std_similarity_incorrect': float(std_incorrect),
}

# Lưu metrics vào file JSON
with open(os.path.join(CURRENT_EVAL_DIR, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=4, default=convert_to_serializable)

# 5.6 Tạo báo cáo tổng hợp
report_content = f"""# Face Recognition Evaluation Report

## 1. Overall Performance
- Accuracy: {accuracy*100:.2f}%
- Average Similarity Score (Correct): {np.mean(correct_scores):.4f} ± {np.std(correct_scores):.4f}
- Average Similarity Score (Incorrect): {np.mean(incorrect_scores):.4f} ± {np.std(incorrect_scores):.4f}

## 2. Per-Class Performance
```
{classification_report(y_true, y_pred, labels=list(database.keys()))}
```

## 3. Visualization
The following visualizations have been generated:
1. Confusion Matrix (confusion_matrix.png)
2. Similarity Score Distribution (similarity_distribution.png)
3. ROC Curves (roc_curves.png)

## 4. Database Statistics
- Number of people in database: {len(database)}
- Average embeddings per person: {np.mean([len(embs) for embs in embeddings_per_person.values()]):.1f}
"""

with open(os.path.join(CURRENT_EVAL_DIR, 'evaluation_report.md'), 'w') as f:
    f.write(report_content)

print(f"\n✅ Đánh giá hoàn tất. Kết quả được lưu trong thư mục: {CURRENT_EVAL_DIR}")
