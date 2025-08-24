# Face Recognition Evaluation Report

## 1. Overall Performance
- Accuracy: 100.00%
- Balanced Accuracy: 100.00%
- Cohen's Kappa: 1.0000

## 2. Rank-K Performance
- Rank-1: 1.0000
- Rank-5: 1.0000

## 3. Verification Performance
- EER: 0.0005
- TAR@FAR=1e-3: 0.9997
- TAR@FAR=1e-4: 0.9995

## 4. Per-Class Performance
```
                   precision    recall  f1-score   support

     DOAN_HUU_DUC       1.00      1.00      1.00        85
 NGUYEN_GIA_DUONG       1.00      1.00      1.00        71
  NGUYEN_KE_LUONG       1.00      1.00      1.00        73
NGUYEN_THI_ANH_LY       1.00      1.00      1.00        18
 TRAN_TRONG_KHANG       1.00      1.00      1.00       152
   TRIEU_HUY_MANH       1.00      1.00      1.00        53

         accuracy                           1.00       452
        macro avg       1.00      1.00      1.00       452
     weighted avg       1.00      1.00      1.00       452

```

## 5. Visualization
The following visualizations have been generated:
1. Confusion Matrix (confusion_matrix.png)
2. Verification ROC Curve (verification_roc.png)

## 6. Database Statistics
- Number of people in database: 6
- Average embeddings per person: 302.0
