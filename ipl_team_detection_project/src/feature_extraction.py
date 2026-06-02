import cv2
import numpy as np

from skimage.feature import hog
from skimage.feature import local_binary_pattern

from src.utils import normalize_vector


class FeatureExtractor:

    def __init__(self):
        pass

    # HSV HISTOGRAM
    def extract_hsv_histogram(self, cell_img):

        hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)

        features = []

        for channel in range(3):

            hist = cv2.calcHist(
                [hsv],
                [channel],
                None,
                [16],
                [0, 256]
            )

            hist = cv2.normalize(hist, hist).flatten()

            features.extend(hist)

        return np.array(features)

    # HOG FEATURES
    def extract_hog_features(self, cell_img):

        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)

        gray = cv2.resize(gray, (64, 64))

        features = hog(
            gray,
            orientations=8,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            feature_vector=True
        )

        return features

    # LBP FEATURES
    def extract_lbp_features(self, cell_img):

        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)

        lbp = local_binary_pattern(
            gray,
            P=8,
            R=1,
            method='uniform'
        )

        hist, _ = np.histogram(
            lbp.ravel(),
            bins=10,
            range=(0, 10)
        )

        hist = hist.astype(np.float32)

        hist /= (hist.sum() + 1e-6)

        return hist

    # COMBINED FEATURES
    def extract_features(self, cell_img):

        hsv_features = self.extract_hsv_histogram(cell_img)

        hog_features = self.extract_hog_features(cell_img)

        lbp_features = self.extract_lbp_features(cell_img)

        feature_vector = np.concatenate([
            hsv_features,
            hog_features,
            lbp_features
        ])

        return normalize_vector(feature_vector)