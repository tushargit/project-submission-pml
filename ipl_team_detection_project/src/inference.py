import pickle
import pandas as pd
import numpy as np
from src.utils import load_image, resize_image
from src.grid_split import GridSplitter
from src.feature_extraction import FeatureExtractor


class IPLInference:

    def __init__(self, model_path):

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        self.grid_splitter = GridSplitter()
        self.feature_extractor = FeatureExtractor()

    def predict_image(self, image_path):

        image = load_image(image_path)
        image = resize_image(image)

        cells = self.grid_splitter.split_into_cells(image)

        predictions = []

        for cell in cells:

            features = self.feature_extractor.extract_features(cell)

            # GET CLASS PROBABILITIES
            probabilities = self.model.predict_proba([features])[0]

            # BEST CLASS
            pred = np.argmax(probabilities)

            # CONFIDENCE SCORE
            confidence = np.max(probabilities)

            # THRESHOLDING
            if confidence < 0.45:
                pred = 0

            predictions.append(int(pred))

        return predictions

    def save_predictions(self, image_path, predictions, save_path):

        data = {
            'Image File Name': image_path,
            'Train Or Test': 'Test'
        }

        for idx, pred in enumerate(predictions):
            data[f'c{idx+1:02d}'] = pred

        df = pd.DataFrame([data])
        df.to_csv(save_path, index=False)