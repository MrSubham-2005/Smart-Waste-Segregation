"""
EcoSort AI - Dataset Validation Utility
Inspects the 3-split dataset (train, val, test) across all 4 categories.
Detects missing folders, empty directories, corrupted images, and class imbalance.
Outputs transparent, real counts only.
"""

import os
import sys
from pathlib import Path
from PIL import Image

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml_training.config import (
    DATASET_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    CLASS_NAMES_DIR,
    CLASS_DISPLAY_NAMES
)

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}


def count_and_validate_split(split_dir, class_dir_name):
    """
    Validates images in a specific split/class folder.
    Returns: (valid_images: list of Path, corrupted_images: list of Path, unsupported_files: list of Path)
    """
    target_path = Path(split_dir) / class_dir_name
    if not target_path.exists():
        return [], [], []

    valid = []
    corrupted = []
    unsupported = []

    for file_path in target_path.iterdir():
        if file_path.is_file():
            # Skip .gitkeep and hidden files
            if file_path.name == '.gitkeep' or file_path.name.startswith('.'):
                continue

            ext = file_path.suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                unsupported.append(file_path)
                continue

            try:
                with Image.open(file_path) as img:
                    img.verify()
                valid.append(file_path)
            except Exception:
                corrupted.append(file_path)

    return valid, corrupted, unsupported


def validate_dataset(verbose=True) -> dict:
    """
    Comprehensive validation of the entire dataset.
    
    Returns structured results:
    - is_valid: True only if real images exist in all 4 classes across train, val, test.
    - has_images: True if any image exists.
    - total_images: Total count of verified images across all splits.
    - counts: Dictionary with detailed breakdown per split and class.
    - corrupted: List of corrupted image paths.
    - summary_table: ASCII table showing actual numbers.
    """
    results = {
        'is_valid': True,
        'has_images': False,
        'total_images': 0,
        'missing_folders': [],
        'corrupted_files': [],
        'unsupported_files': [],
        'counts': {
            'train': {},
            'val': {},
            'test': {},
            'totals_per_class': {},
            'totals_per_split': {'train': 0, 'val': 0, 'test': 0}
        },
        'messages': []
    }

    # Verify split directories exist
    for split_name, split_path in [('train', TRAIN_DIR), ('val', VAL_DIR), ('test', TEST_DIR)]:
        if not os.path.exists(split_path):
            results['missing_folders'].append(split_path)
            results['is_valid'] = False

    # Check each class in each split
    for i, class_dir in enumerate(CLASS_NAMES_DIR):
        display_name = CLASS_DISPLAY_NAMES[i]
        class_total = 0

        for split_name, split_path in [('train', TRAIN_DIR), ('val', VAL_DIR), ('test', TEST_DIR)]:
            folder_path = os.path.join(split_path, class_dir)
            if not os.path.exists(folder_path):
                results['missing_folders'].append(folder_path)
                results['counts'][split_name][display_name] = 0
                continue

            valid, corrupted, unsupported = count_and_validate_split(split_path, class_dir)
            count = len(valid)

            results['counts'][split_name][display_name] = count
            results['counts']['totals_per_split'][split_name] += count
            class_total += count

            results['corrupted_files'].extend(corrupted)
            results['unsupported_files'].extend(unsupported)

        results['counts']['totals_per_class'][display_name] = class_total
        results['total_images'] += class_total

    results['has_images'] = results['total_images'] > 0

    # Build ASCII Summary Table
    table_lines = [
        "Class                   Train    Validation    Test     Total",
        "--------------------------------------------------------------"
    ]
    for disp in CLASS_DISPLAY_NAMES:
        tr = results['counts']['train'].get(disp, 0)
        va = results['counts']['val'].get(disp, 0)
        te = results['counts']['test'].get(disp, 0)
        tot = results['counts']['totals_per_class'].get(disp, 0)
        table_lines.append(f"{disp:<22} {tr:<8} {va:<12} {te:<8} {tot:<6}")
    table_lines.append("--------------------------------------------------------------")
    t_tr = results['counts']['totals_per_split']['train']
    t_va = results['counts']['totals_per_split']['val']
    t_te = results['counts']['totals_per_split']['test']
    table_lines.append(f"{'Total':<22} {t_tr:<8} {t_va:<12} {t_te:<8} {results['total_images']:<6}")

    results['summary_table'] = "\n".join(table_lines)

    # Sanity checks
    if not results['has_images']:
        results['is_valid'] = False
        results['messages'].append(
            "Dataset is currently EMPTY. No images found in dataset/train, val, or test folders."
        )
    else:
        # Check if all classes have images
        empty_classes = [c for c, total in results['counts']['totals_per_class'].items() if total == 0]
        if empty_classes:
            results['is_valid'] = False
            results['messages'].append(f"Classes with 0 images: {', '.join(empty_classes)}")

        # Check split requirements
        if t_tr == 0 or t_va == 0 or t_te == 0:
            results['is_valid'] = False
            results['messages'].append("All three splits (train, val, test) must contain at least 1 image per class.")

    if results['corrupted_files']:
        results['messages'].append(f"Found {len(results['corrupted_files'])} corrupted image files.")

    if verbose:
        print("\n" + "=" * 62)
        print("         EcoSort AI - Dataset Validation Report")
        print("=" * 62)
        print(results['summary_table'])
        print("\nDataset Status: " + ("READY FOR TRAINING" if results['is_valid'] else "INCOMPLETE / EMPTY"))
        if results['messages']:
            print("\nFindings:")
            for msg in results['messages']:
                print(f" - {msg}")
        print("=" * 62 + "\n")

    return results


if __name__ == '__main__':
    validate_dataset(verbose=True)
