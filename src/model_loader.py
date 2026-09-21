from pathlib import Path
import json
import joblib


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "champion_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
FEATURE_NAMES_PATH = PROJECT_ROOT / "models" / "feature_names.json"


def load_model():
    """Load the trained V2 champion model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def load_metadata():
    """Load V2 model metadata."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_PATH}"
        )

    with open(METADATA_PATH, "r") as f:
        return json.load(f)


def load_feature_names():
    """Load the exact feature schema used during training."""
    if not FEATURE_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Feature schema not found: {FEATURE_NAMES_PATH}"
        )

    with open(FEATURE_NAMES_PATH, "r") as f:
        return json.load(f)


model = load_model()
metadata = load_metadata()
feature_names = load_feature_names()

THRESHOLD = metadata["threshold"]
MODEL_NAME = metadata["champion_model"]
TARGET = metadata["target"]
