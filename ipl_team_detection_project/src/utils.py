import logging
from pathlib import Path
import cv2
import numpy as np


def setup_logger(log_file: str = "project.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)



def load_image(image_path: str):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    return image



def resize_image(image, width=800, height=600):
    return cv2.resize(image, (width, height))



def normalize_vector(vector: np.ndarray):
    vector = vector.astype(np.float32)

    if np.sum(vector) == 0:
        return vector

    return vector / np.linalg.norm(vector)