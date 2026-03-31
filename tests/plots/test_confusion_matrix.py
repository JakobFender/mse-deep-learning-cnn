import numpy as np

from src.plots.confusion_matrix import save_confusion_matrix
from src.utils.constants import TEST_PLOT_RESULTS


def test_save_confusion_matrix_creates_file(tmp_path):
    rng = np.random.default_rng(42)
    shared = rng.integers(0, 10, size=1000)
    y_true = np.concat([rng.integers(0, 10, size=1000), shared])
    y_pred = np.concat([rng.integers(0, 10, size=1000), shared])
    save_path = TEST_PLOT_RESULTS / "confusion_matrix.png"

    save_confusion_matrix(y_pred, y_true, save_path)
    print(save_path)

    assert save_path.exists()
    assert save_path.stat().st_size > 0
