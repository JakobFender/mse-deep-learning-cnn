from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.utils.constants import CLASS_NAMES


def save_confusion_matrix(
        y_pred: np.ndarray,
        y_true: np.ndarray,
        save_location: Path | str,
        class_names: list[str] = CLASS_NAMES,
        cmap="Blues"
):
    save_location = Path(save_location)
    save_location.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(save_location)
