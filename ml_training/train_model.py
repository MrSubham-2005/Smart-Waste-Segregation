"""
EcoSort AI - Real Model Training Pipeline
Trains a MobileNetV2 Transfer Learning classifier for 4-class waste segregation:
1. Wet Waste
2. Dry Waste
3. Recyclable Waste
4. E-Waste

Academic Integrity:
- If dataset is empty, training is cleanly suspended without creating fake models.
- Actual model weights, accuracy, and metrics only come from real images.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml_training.config import (
    ROOT_DIR as CFG_ROOT_DIR,
    DATASET_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    MODEL_DIR,
    MODEL_OUTPUT_PATH,
    CLASS_INDICES_PATH,
    RESULTS_DIR,
    CLASS_NAMES_DIR,
    CLASS_DISPLAY_NAMES,
    CLASS_TO_INDEX,
    INDEX_TO_DISPLAY,
    IMAGE_SIZE,
    INPUT_SHAPE,
    NUM_CLASSES,
    BATCH_SIZE,
    EPOCHS_HEAD,
    EPOCHS_FINE_TUNE,
    LEARNING_RATE_HEAD,
    LEARNING_RATE_FINE_TUNE,
    DROPOUT_RATE,
    RANDOM_SEED,
    EARLY_STOPPING_PATIENCE,
    REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR,
    MIN_LR
)
from ml_training.validate_dataset import validate_dataset


def save_class_indices(output_path=CLASS_INDICES_PATH):
    """
    Saves the canonical index-to-class and index-to-display mapping.
    Ensures perfect alignment with ai_engine/categories.py.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mapping = {
        "class_to_index": CLASS_TO_INDEX,
        "index_to_class": {str(v): k for k, v in CLASS_TO_INDEX.items()},
        "index_to_display": {str(k): v for k, v in INDEX_TO_DISPLAY.items()},
        "categories": CLASS_NAMES_DIR,
        "display_names": CLASS_DISPLAY_NAMES,
        "target_size": list(IMAGE_SIZE)
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2)
    print(f"[OK] Class indices mapping saved to: {output_path}")
    return mapping


def get_data_augmentation():
    """
    Constructs Keras data augmentation sequential pipeline.
    Applies real-time affine and pixel transformations to prevent overfitting.
    """
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.10),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomContrast(0.10),
    ], name="waste_augmentation")


def build_mobilenetv2_model(num_classes=NUM_CLASSES, input_shape=INPUT_SHAPE,
                            dropout_rate=DROPOUT_RATE, learning_rate=LEARNING_RATE_HEAD,
                            weights='imagenet'):
    """
    Constructs the MobileNetV2 Transfer Learning Architecture.
    
    1. Base Model: MobileNetV2 pre-trained on ImageNet (weights frozen).
    2. Input layer: accepts (224, 224, 3) normalized in [-1.0, 1.0].
    3. Custom Classification Head:
       - GlobalAveragePooling2D
       - BatchNormalization
       - Dense(256, ReLU, L2 regularized)
       - Dropout(dropout_rate)
       - Dense(4, Softmax)
    """
    import tensorflow as tf

    # Base model pre-trained on ImageNet without top dense classification layer
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    
    # Forward through frozen feature extractor
    x = base_model(inputs, training=False)
    
    # Custom Classification Head
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.BatchNormalization(name="batch_norm")(x)
    x = tf.keras.layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="dense_feature_256"
    )(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="EcoSort_MobileNetV2")

    # Compile with Adam and Categorical Crossentropy
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, base_model


def load_datasets(train_dir=TRAIN_DIR, val_dir=VAL_DIR, test_dir=TEST_DIR,
                  batch_size=BATCH_SIZE, image_size=IMAGE_SIZE):
    """
    Loads images from train, val, test directories using tf.keras.utils.image_dataset_from_directory.
    Applies [-1, 1] normalization and data augmentation.
    """
    import tensorflow as tf

    normalization_layer = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)
    augmentation = get_data_augmentation()

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        label_mode='categorical',
        class_names=CLASS_NAMES_DIR,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=RANDOM_SEED
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        label_mode='categorical',
        class_names=CLASS_NAMES_DIR,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False
    )

    test_ds = None
    if os.path.exists(test_dir):
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            label_mode='categorical',
            class_names=CLASS_NAMES_DIR,
            image_size=image_size,
            batch_size=batch_size,
            shuffle=False
        )

    # Apply data augmentation and normalization on train dataset
    train_ds = train_ds.map(
        lambda x, y: (normalization_layer(augmentation(x, training=True)), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    # Normalize validation dataset
    val_ds = val_ds.map(
        lambda x, y: (normalization_layer(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    if test_ds is not None:
        test_ds = test_ds.map(
            lambda x, y: (normalization_layer(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


def plot_and_save_history(history_head, history_fine, output_dir=RESULTS_DIR):
    """
    Generates and saves the training & validation accuracy and loss curves.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    acc = history_head.history.get('accuracy', [])
    val_acc = history_head.history.get('val_accuracy', [])
    loss = history_head.history.get('loss', [])
    val_loss = history_head.history.get('val_loss', [])

    if history_fine is not None:
        acc += history_fine.history.get('accuracy', [])
        val_acc += history_fine.history.get('val_accuracy', [])
        loss += history_fine.history.get('loss', [])
        val_loss += history_fine.history.get('val_loss', [])

    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    ax1.plot(epochs_range, acc, label='Training Accuracy', color='#10b981', linewidth=2)
    ax1.plot(epochs_range, val_acc, label='Validation Accuracy', color='#3b82f6', linewidth=2)
    ax1.set_title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Loss Plot
    ax2.plot(epochs_range, loss, label='Training Loss', color='#ef4444', linewidth=2)
    ax2.plot(epochs_range, val_loss, label='Validation Loss', color='#f59e0b', linewidth=2)
    ax2.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.5)

    plot_path = os.path.join(output_dir, 'training_curves.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[OK] Training curves saved to: {plot_path}")

    # Save numeric history
    history_json_path = os.path.join(output_dir, 'training_history.json')
    history_data = {
        'accuracy': [float(x) for x in acc],
        'val_accuracy': [float(x) for x in val_acc],
        'loss': [float(x) for x in loss],
        'val_loss': [float(x) for x in val_loss],
        'total_epochs': len(acc)
    }
    with open(history_json_path, 'w', encoding='utf-8') as f:
        json.dump(history_data, f, indent=2)
    print(f"[OK] Training history metrics saved to: {history_json_path}")


def train_model(epochs_head=EPOCHS_HEAD, epochs_fine_tune=EPOCHS_FINE_TUNE,
                batch_size=BATCH_SIZE, fine_tune=True):
    """
    Main training execution function.
    Validates dataset before running.
    Strictly halts if dataset is empty to uphold academic integrity.
    """
    print("\n" + "=" * 64)
    print("      EcoSort AI - MobileNetV2 Waste Classifier Training")
    print("=" * 64)
    print(f"Target Classes: {', '.join(CLASS_DISPLAY_NAMES)}")
    print(f"Architecture:   MobileNetV2 (Transfer Learning)")
    print(f"Target Output:  {MODEL_OUTPUT_PATH}")
    print("=" * 64)

    # Step 1: Pre-flight Dataset Validation
    val_report = validate_dataset(verbose=True)

    if not val_report['is_valid']:
        print("\n" + "!" * 64)
        print(" [ACADEMIC INTEGRITY] TRAINING SUSPENDED - NO REAL DATASET DETECTED")
        print("!" * 64)
        print("EcoSort AI strictly enforces honest academic reporting.")
        print("No synthetic/fake model file (.keras) will be created.")
        print("\nTo train the real model:")
        print("1. Collect real waste images for all 4 categories:")
        for cat in CLASS_NAMES_DIR:
            print(f"   - dataset/train/{cat}/")
            print(f"   - dataset/val/{cat}/")
            print(f"   - dataset/test/{cat}/")
        print("\n2. Recommended dataset size for good generalization:")
        print("   - At least 100-300 images per class in train/")
        print("   - At least 30-50 images per class in val/")
        print("   - At least 30-50 images per class in test/")
        print("\n3. Re-run training once real images are added:")
        print("   python ml_training/train_model.py")
        print("!" * 64 + "\n")
        return False

    # Step 2: Attempt TensorFlow / Keras Import
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(RANDOM_SEED)
    except Exception as e:
        print("\n[ERROR] TensorFlow initialization failed:")
        print(f"Details: {str(e)}")
        print("\nNote: On Windows systems with Application Control (WDAC / Smart App Control),")
        print("Python wheels must be signed or run within WSL2 / Linux container.")
        return False

    # Step 3: Ensure Output Directories Exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save class indices
    save_class_indices()

    # Step 4: Load Real Datasets
    print("\n[INFO] Loading datasets from disk...")
    train_ds, val_ds, test_ds = load_datasets(batch_size=batch_size)

    # Step 5: Build Model
    print("\n[INFO] Building MobileNetV2 Transfer Learning Architecture...")
    model, base_model = build_mobilenetv2_model()
    model.summary()

    # Step 6: Setup Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LR,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_OUTPUT_PATH,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    # Step 7: Phase 1 - Train Classification Head
    print(f"\n[INFO] Phase 1 Training: Feature Head ({epochs_head} epochs, frozen base)...")
    start_time = time.time()
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_head,
        callbacks=callbacks
    )

    history_fine = None

    # Step 8: Phase 2 - Optional Fine-Tuning
    if fine_tune and epochs_fine_tune > 0:
        print("\n[INFO] Phase 2 Training: Fine-tuning top layers of MobileNetV2...")
        base_model.trainable = True
        # Fine-tune only top 30 layers
        for layer in base_model.layers[:-30]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINE_TUNE),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        fine_tune_callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=MODEL_OUTPUT_PATH,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]

        history_fine = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_fine_tune,
            callbacks=fine_tune_callbacks
        )

    duration = time.time() - start_time
    print(f"\n[OK] Training completed in {duration:.1f} seconds.")

    # Save final model explicitly
    model.save(MODEL_OUTPUT_PATH)
    print(f"[OK] Trained model successfully saved to: {MODEL_OUTPUT_PATH}")

    # Step 9: Save Visualizations and History
    plot_and_save_history(history_head, history_fine)

    print("\n" + "=" * 64)
    print("  EcoSort AI - MobileNetV2 Model Successfully Trained & Saved")
    print("=" * 64)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EcoSort AI - Model Training Pipeline")
    parser.add_argument('--epochs-head', type=int, default=EPOCHS_HEAD, help="Epochs for training classification head")
    parser.add_argument('--epochs-fine', type=int, default=EPOCHS_FINE_TUNE, help="Epochs for fine-tuning")
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument('--no-fine-tune', action='store_true', help="Disable fine-tuning phase")
    args = parser.parse_args()

    train_model(
        epochs_head=args.epochs_head,
        epochs_fine_tune=args.epochs_fine,
        batch_size=args.batch_size,
        fine_tune=not args.no_fine_tune
    )
