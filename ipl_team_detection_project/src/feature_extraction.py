import cv2
import numpy as np

from skimage.feature import hog
from skimage.feature import local_binary_pattern

from src.utils import normalize_vector
from sklearn.cluster import KMeans


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

        dominant_colors = self.extract_dominant_colors(
            cell_img
        )

        color_ratios = self.extract_color_ratios(
            cell_img
        )

        orb_features = self.extract_orb_features(
            cell_img
        )

        feature_vector = np.concatenate([
            hsv_features,
            hog_features,
            lbp_features,
            dominant_colors,
            color_ratios,
            orb_features
        ])

        return normalize_vector(feature_vector)
    def extract_dominant_colors(self, cell_img):
        #Add Dominant Color Features (Highest ROI)
        img = cv2.resize(cell_img, (32, 32))

        pixels = img.reshape(-1, 3)

        kmeans = KMeans(
            n_clusters=3,
            random_state=42,
            n_init=10
        )

        kmeans.fit(pixels)

        dominant_colors = kmeans.cluster_centers_.flatten()

        return dominant_colors
    def extract_color_ratios(self, cell_img):
        
        #Add Color Ratio Feature

        hsv = cv2.cvtColor(
            cell_img,
            cv2.COLOR_BGR2HSV
        )

        h = hsv[:, :, 0]

        blue = np.mean((h >= 90) & (h <= 130))

        red1 = np.mean((h >= 0) & (h <= 10))
        red2 = np.mean((h >= 170) & (h <= 180))
        red = red1 + red2

        yellow = np.mean((h >= 20) & (h <= 35))

        orange = np.mean((h >= 10) & (h <= 20))

        return np.array([
            blue,
            red,
            yellow,
            orange
        ])
    def extract_orb_features(self, cell_img):
        #Add ORB Features
        gray = cv2.cvtColor(
            cell_img,
            cv2.COLOR_BGR2GRAY
        )

        orb = cv2.ORB_create(
            nfeatures=32
        )

        keypoints, descriptors = orb.detectAndCompute(
            gray,
            None
        )

        if descriptors is None:

            return np.zeros(32)

        descriptor_mean = descriptors.mean(axis=0)

        return descriptor_mean