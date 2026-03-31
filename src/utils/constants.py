from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_FOLDER = PROJECT_ROOT / "data"

TRAIN_SET_FOLDER = DATA_FOLDER / "train"
VALIDATE_SET_FOLDER = DATA_FOLDER / "validate"

OUTPUT_FOLDER = PROJECT_ROOT / "out"
TEST_PLOT_RESULTS = OUTPUT_FOLDER / "tests"

CLASS_NAMES = ("cat", "chicken", "cow", "dog", "elephant", "horse", "rabbit", "sheep", "squirrel", "zebra")
