from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_FOLDER = PROJECT_ROOT / "data"

TRAIN_SET_FOLDER = DATA_FOLDER / "train"
VALIDATE_SET_FOLDER = DATA_FOLDER / "validate"
