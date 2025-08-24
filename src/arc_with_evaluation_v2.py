import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import cv2
import numpy as np
import pickle
from tqdm import tqdm
from insightface.app import FaceAnalysis
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                           balanced_accuracy_score, cohen_kappa_score,
                           roc_curve, auc, precision_recall_curve)
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
    emb = faces[0].embedding
    # L2-normalize embedding vector
    return emb / (np.linalg.norm(emb) + 1e-12)

# =====================
# 2. Tạo và lưu database
# =====================
DATASET_TRAIN = "../FRAME_DATASET/train"
database = {}
embeddings_per_person = {}

# Cố định thứ tự label
people = sorted([d for d in os.listdir(DATASET_TRAIN) 
                if os.path.isdir(os.path.join(DATASET_TRAIN, d))])

for person in people:
    folder = os.path.join(DATASET_TRAIN, person)
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
        proto = np.mean(embeddings, axis=0)
        # L2-normalize prototype
        proto = proto / (np.linalg.norm(proto) + 1e-12)
        database[person] = proto
        embeddings_per_person[person] = embeddings

print("✅ Đã tạo database cho:", list(database.keys()))

# Lưu database trong thư mục đánh giá
with open(os.path.join(CURRENT_EVAL_DIR, "face_database.pkl"), "wb") as f:
    pickle.dump(database, f)

# =====================
# 3. Đánh giá trên tập test
# =====================
DATASET_TEST = "../FRAME_DATASET/test"

y_true = []
y_pred = []
y_scores = []
all_distances = []

for person in people:  # Sử dụng thứ tự đã sắp xếp
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

        distances = {name: float(np.dot(emb, db_emb)) 
                    for name, db_emb in database.items()}
        pred_name = max(distances.items(), key=lambda x: x[1])[0]
        pred_score = distances[pred_name]

        y_true.append(person)
        y_pred.append(pred_name)
        y_scores.append(pred_score)
        all_distances.append(distances)

# =====================
# 4. Tính các metric bổ sung
# =====================

# 4.1 Rank-K Accuracy
def calculate_rank_k_accuracy(all_distances, y_true, k_values=[1, 5]):
    results = {}
    n = len(y_true)
    for k in k_values:
        correct = 0
        for distances, true_label in zip(all_distances, y_true):
            # Sắp xếp predictions theo score
            ordered_preds = sorted(distances.items(), key=lambda x: x[1], reverse=True)
            top_k_preds = [name for name, _ in ordered_preds[:k]]
            if true_label in top_k_preds:
                correct += 1
        results[f'rank_{k}'] = correct / n
    return results

# 4.2 Verification Metrics (EER & TAR@FAR)
def calculate_verification_metrics(embeddings_per_person):
    genuine, impostor = [], []
    names = list(embeddings_per_person.keys())
    
    for i, ni in enumerate(names):
        vi = embeddings_per_person[ni]
        for j, nj in enumerate(names):
            vj = embeddings_per_person[nj]
            if i == j:
                for a in range(len(vi)):
                    for b in range(a+1, len(vi)):
                        s = float(np.dot(vi[a], vi[b]))
                        genuine.append(s)
            else:
                # Sample pairs for efficiency
                for a in vi[:min(len(vi), 20)]:
                    for b in vj[:min(len(vj), 20)]:
                        s = float(np.dot(a, b))
                        impostor.append(s)

    scores = np.array(genuine + impostor)
    labels = np.array([1]*len(genuine) + [0]*len(impostor))

    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

    def tar_at_far(target_far):
        idx = np.searchsorted(fpr, target_far, side='left')
        idx = min(idx, len(tpr)-1)
        return tpr[idx]

    metrics = {
        'eer': float(eer),
        'tar_far_1e3': float(tar_at_far(1e-3)),
        'tar_far_1e4': float(tar_at_far(1e-4))
    }
    return metrics, (fpr, tpr, thr)

# =====================
# 5. Tính toán và lưu kết quả
# =====================

# 5.1 Metrics cơ bản
accuracy = accuracy_score(y_true, y_pred)
balanced_acc = balanced_accuracy_score(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred, labels=people)
class_report = classification_report(y_true, y_pred, labels=people, output_dict=True)

# 5.2 Rank-K accuracy
rank_metrics = calculate_rank_k_accuracy(all_distances, y_true)

# 5.3 Verification metrics
ver_metrics, ver_curves = calculate_verification_metrics(embeddings_per_person)

# 5.4 Lưu tất cả metrics
metrics = {
    'accuracy': float(accuracy),
    'balanced_accuracy': float(balanced_acc),
    'cohen_kappa': float(kappa),
    'rank_1': rank_metrics['rank_1'],
    'rank_5': rank_metrics['rank_5'],
    'eer': ver_metrics['eer'],
    'tar_far_1e3': ver_metrics['tar_far_1e3'],
    'tar_far_1e4': ver_metrics['tar_far_1e4'],
    'classification_report': class_report
}

with open(os.path.join(CURRENT_EVAL_DIR, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=4)

# =====================
# 6. Vẽ đồ thị
# =====================

# 6.1 Confusion Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=people, yticklabels=people)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(CURRENT_EVAL_DIR, 'confusion_matrix.png'))
plt.close()

# 6.2 ROC Curve
fpr, tpr, _ = ver_curves
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, 'b-', label=f'ROC (EER = {ver_metrics["eer"]:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Verification ROC Curve')
plt.legend(loc="lower right")
plt.savefig(os.path.join(CURRENT_EVAL_DIR, 'verification_roc.png'))
plt.close()

# Tạo báo cáo markdown
report_content = f"""# Face Recognition Evaluation Report

## 1. Overall Performance
- Accuracy: {accuracy*100:.2f}%
- Balanced Accuracy: {balanced_acc*100:.2f}%
- Cohen's Kappa: {kappa:.4f}

## 2. Rank-K Performance
- Rank-1: {rank_metrics['rank_1']:.4f}
- Rank-5: {rank_metrics['rank_5']:.4f}

## 3. Verification Performance
- EER: {ver_metrics['eer']:.4f}
- TAR@FAR=1e-3: {ver_metrics['tar_far_1e3']:.4f}
- TAR@FAR=1e-4: {ver_metrics['tar_far_1e4']:.4f}

## 4. Per-Class Performance
```
{classification_report(y_true, y_pred, labels=people)}
```

## 5. Visualization
The following visualizations have been generated:
1. Confusion Matrix (confusion_matrix.png)
2. Verification ROC Curve (verification_roc.png)

## 6. Database Statistics
- Number of people in database: {len(database)}
- Average embeddings per person: {np.mean([len(embs) for embs in embeddings_per_person.values()]):.1f}
"""

with open(os.path.join(CURRENT_EVAL_DIR, 'evaluation_report.md'), 'w') as f:
    f.write(report_content)

print(f"\n✅ Đánh giá hoàn tất. Kết quả được lưu trong thư mục: {CURRENT_EVAL_DIR}")
