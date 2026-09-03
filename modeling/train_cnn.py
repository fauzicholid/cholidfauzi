#!/usr/bin/env python3
"""Train the LULC CNN on patches produced by prepare_training_patches.py.

Usage:
    python train_cnn.py --patches training_patches.npz --model-dir runs/lulc_v1
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

from cnn_model import build_cnn


def load_dataset(patches_path: Path):
    data = np.load(patches_path, allow_pickle=True)
    X, y = data["X"], data["y"]
    classes = data["classes"]
    class_labels = data["class_labels"]
    band_mean, band_std = data["band_mean"], data["band_std"]
    patch_size = int(data["patch_size"])
    return X, y, classes, class_labels, band_mean, band_std, patch_size


def normalize(X: np.ndarray, band_mean: np.ndarray, band_std: np.ndarray) -> np.ndarray:
    return (X - band_mean) / band_std


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches", required=True, type=Path, help=".npz file from prepare_training_patches.py")
    parser.add_argument("--model-dir", required=True, type=Path, help="Directory to write the trained model and metadata")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of samples held out for validation (default: 0.2)")
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size (default: 64)")
    parser.add_argument("--patience", type=int, default=8, help="Early-stopping patience on val_loss (default: 8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the train/val split")
    return parser.parse_args()


def main():
    args = parse_args()
    X, y, classes, class_labels, band_mean, band_std, patch_size = load_dataset(args.patches)

    if len(np.unique(y)) < 2:
        raise ValueError("Need at least 2 classes to train a classifier")

    X = normalize(X, band_mean, band_std)
    n_bands = X.shape[-1]
    n_classes = len(classes)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_split, random_state=args.seed, stratify=y
    )

    model = build_cnn(patch_size=patch_size, n_bands=n_bands, n_classes=n_classes)
    model.summary()

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / "lulc_cnn.keras"

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Final validation accuracy: {val_acc:.4f} (loss: {val_loss:.4f})")

    metadata = {
        "classes": [int(c) for c in classes],
        "class_labels": [str(c) for c in class_labels],
        "band_mean": band_mean.tolist(),
        "band_std": band_std.tolist(),
        "patch_size": patch_size,
        "n_bands": int(n_bands),
        "val_accuracy": float(val_acc),
        "val_loss": float(val_loss),
    }
    with open(args.model_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    history_path = args.model_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

    print(f"Saved model -> {model_path}")
    print(f"Saved metadata -> {args.model_dir / 'model_metadata.json'}")
    print(f"Saved training history -> {history_path}")


if __name__ == "__main__":
    main()
