import logging
import numpy as np
import pandas as pd

from tqdm import tqdm

from src.utils import load_image, resize_image
from src.grid_split import GridSplitter
from src.feature_extraction import FeatureExtractor
from src.config import TRAIN_DIR, TEST_DIR

class DatasetBuilder:

    def __init__(self, csv_path):

        self.csv_path = csv_path

        self.grid_splitter = GridSplitter()
        self.feature_extractor = FeatureExtractor()

    def build_dataset(self):

        df = pd.read_csv(self.csv_path)

        X = []
        y = []

        for _, row in tqdm(df.iterrows(), total=len(df)):
            filename = str(row['image']).strip()

                    # filename from CSV
            filename = str(row['image']).strip()

            # full image path
            image_path = TRAIN_DIR / filename

            print(image_path)


            try:
                image = load_image(image_path)
                image = resize_image(image)

                cells = self.grid_splitter.split_into_cells(image)

                for idx, cell in enumerate(cells):

                    feature_vector = self.feature_extractor.extract_features(cell)

                    label_col = f'c{idx+1:02d}'
                    label = row[label_col]

                    # SKIP MOST EMPTY CELLS
                    if label == 0:

                        # keep only some empty cells
                        import random

                        if random.random() > 0.15:
                            continue

                    X.append(feature_vector)
                    y.append(label)

            except Exception as e:
                logging.error(f"Error processing {image_path}: {e}")

        X = np.array(X)
        y = np.array(y)

        return X, y

    def save_features(self, X, y, save_path='features.npz'):

        np.savez_compressed(save_path, X=X, y=y)