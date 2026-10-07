"""Classification metrics shared by every model."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def compute_metrics(y_true, y_prob, threshold=0.5):
    """Fake (label 1) is the positive class, so precision and recall are for fake detection."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        # rows are the true class, columns the predicted class, order [real, fake]
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def save_metrics(path, metrics, model_name, params, train_seconds, ms_per_image,
                 img_size, seed, n_train_images):
    result = {
        "model": model_name,
        **metrics,
        "trainable_params": params,
        "train_seconds": train_seconds,
        "ms_per_image": ms_per_image,
        "img_size": img_size,
        "seed": seed,
        "n_train_images": n_train_images,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
