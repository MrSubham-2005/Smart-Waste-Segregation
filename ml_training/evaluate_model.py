"""
EcoSort AI - Real Model Evaluation Pipeline
Evaluates the trained MobileNetV2 classifier on the test split (dataset/test/).
Calculates:
- Test Accuracy
- Per-class Precision, Recall, F1-score
- Macro and Weighted averages
- Confusion Matrix (4x4)

Academic Integrity:
- If no trained model or test images exist, cleanly suspends without generating fake metrics.
- Uses pure NumPy fallback for metrics calculation so evaluation never fails on binary-restricted environments.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml_training.config import (
    ROOT_DIR as CFG_ROOT_DIR,
    TEST_DIR,
    MODEL_OUTPUT_PATH,
    CLASS_INDICES_PATH,
    RESULTS_DIR,
    CLASS_NAMES_DIR,
    CLASS_DISPLAY_NAMES,
    INDEX_TO_DISPLAY,
    IMAGE_SIZE,
    BATCH_SIZE
)
from ml_training.validate_dataset import count_and_validate_split


def compute_metrics_numpy(y_true, y_pred, num_classes=4):
    """
    Pure NumPy implementation of classification metrics and confusion matrix.
    Guarantees calculation runs reliably without depending on compiled C++ extensions.
    
    :param y_true: 1D array of ground truth class indices (0 to num_classes-1)
    :param y_pred: 1D array of predicted class indices (0 to num_classes-1)
    :return: dict with accuracy, precision, recall, f1 per class, averages, and confusion matrix
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    total_samples = len(y_true)

    if total_samples == 0:
        return None

    # Confusion Matrix: rows = true, cols = predicted
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    accuracy = float(np.sum(np.diag(cm)) / total_samples)

    per_class = {}
    precisions = []
    recalls = []
    f1s = []
    supports = []

    for c in range(num_classes):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        support = int(np.sum(cm[c, :]))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

        per_class[CLASS_DISPLAY_NAMES[c]] = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'support': support
        }

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    total_support = sum(supports)
    if total_support > 0:
        weighted_precision = float(sum(p * s for p, s in zip(precisions, supports)) / total_support)
        weighted_recall = float(sum(r * s for r, s in zip(recalls, supports)) / total_support)
        weighted_f1 = float(sum(f * s for f, s in zip(f1s, supports)) / total_support)
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0

    return {
        'total_samples': total_samples,
        'accuracy': round(accuracy, 4),
        'per_class': per_class,
        'macro_avg': {
            'precision': round(macro_precision, 4),
            'recall': round(macro_recall, 4),
            'f1_score': round(macro_f1, 4)
        },
        'weighted_avg': {
            'precision': round(weighted_precision, 4),
            'recall': round(weighted_recall, 4),
            'f1_score': round(weighted_f1, 4)
        },
        'confusion_matrix': cm.tolist()
    }


def format_classification_report(metrics_dict):
    """
    Formats the metrics dictionary into standard tabular text.
    """
    lines = [
        f"{'Class':<22} {'Precision':<12} {'Recall':<10} {'F1-Score':<10} {'Support':<8}",
        "-" * 64
    ]
    for name, m in metrics_dict['per_class'].items():
        lines.append(
            f"{name:<22} {m['precision']:<12.4f} {m['recall']:<10.4f} {m['f1_score']:<10.4f} {m['support']:<8}"
        )
    lines.append("-" * 64)
    lines.append(f"{'Accuracy':<46} {metrics_dict['accuracy']:<10.4f} {metrics_dict['total_samples']:<8}")

    m_macro = metrics_dict['macro_avg']
    lines.append(
        f"{'Macro Avg':<22} {m_macro['precision']:<12.4f} {m_macro['recall']:<10.4f} {m_macro['f1_score']:<10.4f} {metrics_dict['total_samples']:<8}"
    )

    m_weight = metrics_dict['weighted_avg']
    lines.append(
        f"{'Weighted Avg':<22} {m_weight['precision']:<12.4f} {m_weight['recall']:<10.4f} {m_weight['f1_score']:<10.4f} {metrics_dict['total_samples']:<8}"
    )
    return "\n".join(lines)


def format_confusion_matrix_ascii(cm_list):
    """
    Formats the confusion matrix as a clear ASCII table.
    """
    header = f"{'True \\ Pred':<20} " + " ".join(f"{c[:10]:>12}" for c in CLASS_DISPLAY_NAMES)
    sep = "-" * len(header)
    lines = [header, sep]
    for i, row in enumerate(cm_list):
        row_str = f"{CLASS_DISPLAY_NAMES[i]:<20} " + " ".join(f"{val:>12}" for val in row)
        lines.append(row_str)
    return "\n".join(lines)


def save_confusion_matrix_plot(cm_list, output_path):
    """
    Renders and saves a visual confusion matrix heatmap using matplotlib.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        cm = np.array(cm_list)
        fig, ax = plt.subplots(figsize=(8, 7))
        cax = ax.matshow(cm, cmap='Blues')
        fig.colorbar(cax)

        ax.set_xticks(range(len(CLASS_DISPLAY_NAMES)))
        ax.set_yticks(range(len(CLASS_DISPLAY_NAMES)))
        ax.set_xticklabels(CLASS_DISPLAY_NAMES, rotation=30, ha='left')
        ax.set_yticklabels(CLASS_DISPLAY_NAMES)

        # Annotate cells
        for i in range(len(CLASS_DISPLAY_NAMES)):
            for j in range(len(CLASS_DISPLAY_NAMES)):
                val = cm[i, j]
                color = 'white' if val > (cm.max() / 2) else 'black'
                ax.text(j, i, str(val), va='center', ha='center', color=color, fontsize=12, fontweight='bold')

        plt.title('EcoSort AI - Waste Classification Confusion Matrix', pad=20, fontsize=13, fontweight='bold')
        plt.xlabel('Predicted Class', fontsize=11, labelpad=10)
        plt.ylabel('Actual (Ground Truth) Class', fontsize=11, labelpad=10)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"[OK] Confusion matrix plot saved to: {output_path}")
    except Exception as e:
        print(f"[WARN] Could not save confusion matrix plot: {e}")


def evaluate_model(model_path=MODEL_OUTPUT_PATH, test_dir=TEST_DIR):
    """
    Main evaluation routine.
    Validates model existence and test images before evaluating.
    """
    print("\n" + "=" * 64)
    print("      EcoSort AI - MobileNetV2 Model Evaluation")
    print("=" * 64)
    print(f"Target Model:  {model_path}")
    print(f"Test Split:    {test_dir}")
    print("=" * 64)

    # Pre-check 1: Check if trained model file exists
    if not os.path.exists(model_path):
        print("\n" + "!" * 64)
        print(" [ACADEMIC INTEGRITY] EVALUATION SUSPENDED - NO TRAINED MODEL FOUND")
        print("!" * 64)
        print(f"Model file '{model_path}' does not exist.")
        print("Evaluation cannot proceed without a genuine trained model file.")
        print("In accordance with AICTE academic integrity principles:")
        print(" - No fake test accuracy or synthetic precision scores will be reported.")
        print(" - The model must first be trained using real images via:")
        print("   python ml_training/train_model.py")
        print("!" * 64 + "\n")
        return None

    # Pre-check 2: Check test split images
    total_test_images = 0
    test_counts = {}
    for cat in CLASS_NAMES_DIR:
        valid, _, _ = count_and_validate_split(test_dir, cat)
        cnt = len(valid)
        test_counts[cat] = cnt
        total_test_images += cnt

    if total_test_images == 0:
        print("\n" + "!" * 64)
        print(" [ACADEMIC INTEGRITY] EVALUATION SUSPENDED - TEST DATASET IS EMPTY")
        print("!" * 64)
        print(f"No test images found in '{test_dir}'.")
        print("Please place real test images in dataset/test/ to evaluate.")
        print("!" * 64 + "\n")
        return None

    # Step 3: Load Model & Run Evaluation
    try:
        import tensorflow as tf
    except Exception as e:
        print(f"[ERROR] TensorFlow import failed: {e}")
        return None

    print("\n[INFO] Loading trained Keras model...")
    model = tf.keras.models.load_model(model_path)

    print(f"[INFO] Evaluating across {total_test_images} test images...")
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        label_mode='int',
        class_names=CLASS_NAMES_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    y_true_list = []
    y_pred_list = []

    for images, labels in test_ds:
        normalized_imgs = normalization_layer(images)
        preds = model.predict(normalized_imgs, verbose=0)
        pred_classes = np.argmax(preds, axis=1)

        y_true_list.extend(labels.numpy().tolist())
        y_pred_list.extend(pred_classes.tolist())

    # Step 4: Compute Metrics
    metrics = compute_metrics_numpy(y_true_list, y_pred_list, num_classes=len(CLASS_NAMES_DIR))

    # Step 5: Save Results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_text = format_classification_report(metrics)
    cm_ascii = format_confusion_matrix_ascii(metrics['confusion_matrix'])

    json_path = os.path.join(RESULTS_DIR, 'evaluation_metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    txt_path = os.path.join(RESULTS_DIR, 'classification_report.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("EcoSort AI - Classification Report\n")
        f.write("=" * 64 + "\n\n")
        f.write(report_text + "\n\n")
        f.write("Confusion Matrix:\n")
        f.write(cm_ascii + "\n")

    plot_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    save_confusion_matrix_plot(metrics['confusion_matrix'], plot_path)

    print("\n" + "=" * 64)
    print("         EcoSort AI - Classification Report")
    print("=" * 64)
    print(report_text)
    print("\n" + "=" * 64)
    print("         EcoSort AI - Confusion Matrix")
    print("=" * 64)
    print(cm_ascii)
    print("\n" + "=" * 64)
    print(f"[OK] Evaluation results saved to {RESULTS_DIR}")
    print("=" * 64 + "\n")

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EcoSort AI - Model Evaluation")
    parser.add_argument('--model-path', type=str, default=MODEL_OUTPUT_PATH, help="Path to .keras model")
    parser.add_argument('--test-dir', type=str, default=TEST_DIR, help="Path to test dataset")
    args = parser.parse_args()

    evaluate_model(model_path=args.model_path, test_dir=args.test_dir)
