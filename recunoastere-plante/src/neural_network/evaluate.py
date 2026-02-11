# src/neural_network/evaluate.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
TEST_DIR = DATA_DIR / "test"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_SCREEN_DIR = PROJECT_ROOT / "docs" / "screenshots"

IMG_SIZE = (150, 150)
BATCH_SIZE = 8


def main():
    DOCS_SCREEN_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "trained_model.h5"
    if not model_path.exists():
        # fallback
        model_path = MODELS_DIR / "trained_model_final.h5"
    if not model_path.exists():
        raise FileNotFoundError("Nu găsesc modelul antrenat în models/ (trained_model.h5 sau trained_model_final.h5).")

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )
    class_names = test_ds.class_names

    model = tf.keras.models.load_model(model_path)

    # verificare: modelul trebuie să aibă același număr de ieșiri ca numărul de clase din test
    out_units = model.output_shape[-1]
    if out_units != len(class_names):
        raise ValueError(
            f"Mismtach clase: model outputs={out_units}, test classes={len(class_names)} ({class_names}). "
            "Asigură-te că test-ul are aceleași clase ca train-ul."
        )

    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
    y_prob = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    cm = confusion_matrix(y_true, y_pred)

    # report text
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    (MODELS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    print(report)

    # plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # valori în celule
    thresh = cm.max() * 0.5 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.tight_layout()
    out_path = DOCS_SCREEN_DIR / "confusion_matrix.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    # salvează cm și în json (util)
    with open(MODELS_DIR / "confusion_matrix.json", "w", encoding="utf-8") as f:
        json.dump({"classes": class_names, "matrix": cm.tolist()}, f, ensure_ascii=False, indent=2)

    print(f"[OK] Confusion matrix salvată în: {out_path}")


if __name__ == "__main__":
    main()