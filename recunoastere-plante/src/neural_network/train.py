# src/neural_network/train.py
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

import tensorflow as tf

from .model import build_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

MODELS_DIR = PROJECT_ROOT / "models"
DOCS_SCREEN_DIR = PROJECT_ROOT / "docs" / "screenshots"

IMG_SIZE = (150, 150)
BATCH_SIZE = 2  # dataset mic -> batch mic ajută
SEED = 44

# split train/val/test
SPLIT_TRAIN_VAL_TEST = (0.8, 0.1, 0.1)  # train, val, test


def _list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def _has_any_images(folder: Path) -> bool:
    return folder.exists() and len(_list_images(folder)) > 0


def prepare_splits_train_val_test(
    processed_dir: Path,
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
    split: Tuple[float, float, float] = SPLIT_TRAIN_VAL_TEST,
    seed: int = SEED,
) -> None:
    """
    Creează folderele data/train, data/val și data/test din data/processed păstrând subfolderele (clasele).
    Nu reface split-ul dacă train/val/test au deja imagini.
    """
    if _has_any_images(train_dir) or _has_any_images(val_dir) or _has_any_images(test_dir):
        print("[INFO] train/val/test conțin deja imagini. Nu refac split-ul.")
        return

    if not processed_dir.exists():
        raise FileNotFoundError(f"Nu există folderul: {processed_dir}")

    classes = [d for d in processed_dir.iterdir() if d.is_dir()]
    if not classes:
        raise ValueError(f"Nu am găsit clase în {processed_dir} (aștept subfoldere).")

    tr, va, te = split
    if abs((tr + va + te) - 1.0) > 1e-6:
        raise ValueError(f"Split-ul trebuie să însumeze 1.0, dar este {split}")

    random.seed(seed)
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for cls_dir in classes:
        images = _list_images(cls_dir)
        if not images:
            print(f"[WARN] Clasa '{cls_dir.name}' nu are imagini. Sar peste.")
            continue

        random.shuffle(images)
        n = len(images)

        n_train = int(n * tr)
        n_val = int(n * va)
        n_test = n - n_train - n_val

        # reguli pentru dataset mic:
        if n >= 3:
            if n_val == 0:
                n_val = 1
            if n_test == 0:
                n_test = 1
            n_train = n - n_val - n_test
            if n_train <= 0:
                n_train = 1
                remain = n - n_train
                n_val = max(0, min(n_val, remain))
                n_test = remain - n_val
        elif n == 2:
            # standard: 1 train, 1 test, 0 val
            n_train, n_val, n_test = 1, 0, 1
        else:  # n == 1
            n_train, n_val, n_test = 1, 0, 0

        train_imgs = images[:n_train]
        val_imgs = images[n_train : n_train + n_val]
        test_imgs = images[n_train + n_val :]

        (train_dir / cls_dir.name).mkdir(parents=True, exist_ok=True)
        (val_dir / cls_dir.name).mkdir(parents=True, exist_ok=True)
        (test_dir / cls_dir.name).mkdir(parents=True, exist_ok=True)

        for p in train_imgs:
            shutil.copy2(p, train_dir / cls_dir.name / p.name)
        for p in val_imgs:
            shutil.copy2(p, val_dir / cls_dir.name / p.name)
        for p in test_imgs:
            shutil.copy2(p, test_dir / cls_dir.name / p.name)

        print(
            f"[INFO] Clasa '{cls_dir.name}': total={n}, "
            f"train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}"
        )


def build_datasets_train_val_test(
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
    img_size: Tuple[int, int] = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=True,
        seed=seed,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
    )

    # IMPORTANT: păstrează class_names ÎNAINTE de cache/prefetch
    class_names = train_ds.class_names

    # Verifică să fie aceleași clase peste tot
    if val_ds.class_names != class_names:
        raise ValueError(
            f"Clase diferite între train și val!\n"
            f"train: {class_names}\n"
            f"val:   {val_ds.class_names}"
        )
    if test_ds.class_names != class_names:
        raise ValueError(
            f"Clase diferite între train și test!\n"
            f"train: {class_names}\n"
            f"test:  {test_ds.class_names}"
        )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)
    test_ds = test_ds.cache().prefetch(buffer_size=autotune)

    return train_ds, val_ds, test_ds, class_names


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_SCREEN_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] Pregătesc split train/val/test...")
    prepare_splits_train_val_test(PROCESSED_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR)

    print("[INFO] Construiesc dataset-urile...")
    train_ds, val_ds, test_ds, class_names = build_datasets_train_val_test(TRAIN_DIR, VAL_DIR, TEST_DIR)

    num_classes = len(class_names)
    print(f"[INFO] Clase detectate ({num_classes}): {class_names}")

    # Salvează maparea claselor (utilă la inferență/UI)
    class_map = {name: i for i, name in enumerate(class_names)}
    with open(MODELS_DIR / "class_map.json", "w", encoding="utf-8") as f:
        json.dump(class_map, f, ensure_ascii=False, indent=2)

    model = build_model(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        num_classes=num_classes,
        learning_rate=1e-3,
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / "trained_model.h5"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
        ),
        tf.keras.callbacks.CSVLogger(str(MODELS_DIR / "training_log.csv")),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
        ),
    ]

    print("[INFO] Încep antrenarea (cu validation).")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        callbacks=callbacks,
        verbose=1,
    )

    print("[INFO] Evaluare pe test.")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"[RESULT] Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    # salvează și model final
    model.save(MODELS_DIR / "trained_model_final.h5")

    with open(MODELS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_loss": float(test_loss),
                "test_accuracy": float(test_acc),
                "classes": class_names,
                "img_size": IMG_SIZE,
                "batch_size": BATCH_SIZE,
                "split": SPLIT_TRAIN_VAL_TEST,
                "note": "Train/Val/Test split generat automat din data/processed.",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("[OK] Antrenare terminată. Model salvat în models/.")


if __name__ == "__main__":
    main()