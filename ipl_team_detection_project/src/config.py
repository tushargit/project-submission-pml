from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "dataset" / "train" / "images"
TEST_DIR = BASE_DIR / "dataset" / "test" / "images"

CSV_PATH = BASE_DIR / "master_csv.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"
FEATURE_DIR = BASE_DIR / "features"

MODEL_PATH = MODEL_DIR / "model_team.pkl"

IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600

GRID_ROWS = 8
GRID_COLS = 8

CELL_WIDTH = 100
CELL_HEIGHT = 75

NUM_CLASSES = 11

CLASS_NAMES = {
    0: "No Team",
    1: "CSK",
    2: "DC",
    3: "GT",
    4: "KKR",
    5: "LSG",
    6: "MI",
    7: "PBKS",
    8: "RR",
    9: "RCB",
    10: "SRH"
}

RANDOM_STATE = 42