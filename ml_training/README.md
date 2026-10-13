# EcoSort AI — Machine Learning Training & Evaluation Pipeline

This directory contains the complete training, dataset validation, and evaluation pipeline for the **EcoSort AI** waste segregation model using **Transfer Learning** with **MobileNetV2**.

---

## 1. Target Waste Classes

The classification system is trained to distinguish between four distinct waste streams, mapped to standardized segregation bins:

| Class Key | Display Name | Bin Color | Examples |
|---|---|---|---|
| `wet_waste` | **Wet Waste** | 🟢 Green Bin | Fruit peels, vegetable leftovers, cooked food, coffee grounds, tea bags, garden trimmings |
| `dry_waste` | **Dry Waste** | 🔵 Blue Bin | Non-recyclable plastics, dirty wrappers, napkins, soiled paper, thermocol, broken ceramics |
| `recyclable_waste` | **Recyclable Waste** | 🟡 Yellow Bin (Clean Recyclables) | Clean plastic bottles, paper, cardboard, aluminum cans, glass bottles, scrap metals |
| `e_waste` | **E-Waste** | 🔴 Red Bin (Specialized E-Waste Bin) | Batteries, cables, circuit boards, smartphones, chargers, bulbs, discarded electronic parts |

---

## 2. Dataset Structure

To train and evaluate the model, organize your real-world images according to the 3-split directory hierarchy:

```
dataset/
├── train/
│   ├── wet_waste/
│   ├── dry_waste/
│   ├── recyclable_waste/
│   └── e_waste/
├── val/
│   ├── wet_waste/
│   ├── dry_waste/
│   ├── recyclable_waste/
│   └── e_waste/
└── test/
    ├── wet_waste/
    ├── dry_waste/
    ├── recyclable_waste/
    └── e_waste/
```

### Image Requirements:
- **Formats:** JPG, JPEG, PNG, WEBP.
- **Resolution:** Any standard camera resolution (automatically preprocessed and resized to $224 \times 224$).
- **Color Mode:** RGB (Grayscale and RGBA transparency are converted automatically).
- **Recommended Split Ratios:**
  - **Train:** ~70% (e.g., 100–300+ images per class)
  - **Validation:** ~15% (e.g., 30–50+ images per class)
  - **Test:** ~15% (e.g., 30–50+ images per class)
- **Diversity:** Include varied backgrounds, lighting conditions, angles, and real-world clutter to ensure model robustness.

---

## 3. Step-by-Step Training Workflow

### Step 1: Pre-flight Dataset Validation
Before training, verify your image dataset to identify any corrupted images or class imbalances:

```bash
python ml_training/validate_dataset.py
```

This utility will:
- Count images across all 4 classes in `train`, `val`, and `test` splits.
- Verify image headers and detect corrupted or non-image files.
- Print an ASCII validation table with true image counts.
- Confirm whether the dataset meets the requirements for training.

---

### Step 2: Model Training

Once real images are placed in the dataset directories, execute:

```bash
python ml_training/train_model.py
```

#### Optional Command-Line Arguments:
- `--epochs-head <int>`: Epochs for training the custom classification head (default: `15`).
- `--epochs-fine <int>`: Epochs for fine-tuning the top layers of MobileNetV2 (default: `10`).
- `--batch-size <int>`: Mini-batch size (default: `32`).
- `--no-fine-tune`: Skip fine-tuning and train only the classification head.

Example:
```bash
python ml_training/train_model.py --epochs-head 20 --epochs-fine 10 --batch-size 32
```

#### What Happens During Training:
1. **Pre-flight Validation:** Verifies non-empty dataset. If empty, halts cleanly without creating fake files.
2. **Data Augmentation:** Real-time affine transformations (random flip, rotation, zoom, contrast) prevent overfitting.
3. **Feature Extraction (Phase 1):** The pre-trained MobileNetV2 backbone (trained on ImageNet) remains frozen, training only the custom dense classification head.
4. **Fine-Tuning (Phase 2):** Top 30 layers of MobileNetV2 are unfrozen with a low learning rate ($5 \times 10^{-5}$) to specialize features for waste classification.
5. **Callbacks:**
   - `EarlyStopping` prevents overfitting when validation loss stops improving.
   - `ReduceLROnPlateau` dynamically halves learning rate on plateaus.
   - `ModelCheckpoint` saves only the best performing checkpoint.
6. **Artifact Output:**
   - Saved Keras Model: `ai_engine/models/ecosort_mobilenetv2.keras`
   - Class Mapping: `ai_engine/models/class_indices.json`
   - Training Curves: `ml_training/results/training_curves.png`
   - History Metrics: `ml_training/results/training_history.json`

---

### Step 3: Independent Model Evaluation

Evaluate the trained model strictly on the held-out test split:

```bash
python ml_training/evaluate_model.py
```

This script generates:
- **Accuracy:** Total percentage of correctly classified test images.
- **Classification Report:** Per-class Precision, Recall, and F1-Score.
- **Confusion Matrix:** $4 \times 4$ matrix showing actual vs. predicted classes.
- **Artifacts Saved:**
  - `ml_training/results/classification_report.txt`
  - `ml_training/results/evaluation_metrics.json`
  - `ml_training/results/confusion_matrix.png`

---

## 4. Academic Integrity & Ethics Policy

In strict alignment with AICTE B.Tech CSE guidelines and research ethics:
- **No Synthetic Accuracy:** Accuracy metrics are never hardcoded or fabricated.
- **No Fake Checkpoints:** No `.keras` weights file is generated unless real training occurred on real images.
- **Transparent Fallback:** When the model is not trained, the application reports `status: "model_not_trained"` gracefully without guessing or simulating predictions.
