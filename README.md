# 🩺 Diabetes Risk Classification

A machine-learning workflow for predicting diabetes status using two datasets:
- **db1** → full labels (0 = no diabetes, 1 = type-1, 2 = type-2) - Imbalance
- **db2** → partial labels (0/1) - Balance data - used for testing

The project performs:
1. **Type restoration:** Train on db1 to recreate type-1 / type-2 labels for db2  
2. **Binary stage:** Retrain a second-stage classifier (0 vs 1) to match db2's labeling  
3. **Visualization:** Plot db2 predictions (scatter, bar charts) + confusion matrix  
4. **Evaluation:** Precision, recall, F1, accuracy

---

## 📦 Environment Setup
```bash
conda create -n diabetes python=3.10
conda activate diabetes
pip install -r requirements.txt
