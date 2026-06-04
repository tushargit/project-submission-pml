import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from collections import Counter

from src.config import OUTPUT_DIR


class FeatureAnalyzer:

    def __init__(self, X, y):

        self.X = X
        self.y = y

    # 1. CLASS DISTRIBUTION
    def plot_class_distribution(self):

        counter = Counter(self.y)

        plt.figure(figsize=(12, 6))

        sns.barplot(
            x=list(counter.keys()),
            y=list(counter.values())
        )

        plt.title("Class Distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")

        plt.savefig(
            OUTPUT_DIR / "class_distribution_analysis.png"
        )

        plt.close()

    # 2. FEATURE HISTOGRAM
    def plot_feature_histogram(self):

        plt.figure(figsize=(12, 6))

        plt.hist(
            self.X.flatten(),
            bins=50
        )

        plt.title("Feature Value Distribution")

        plt.savefig(
            OUTPUT_DIR / "feature_histogram.png"
        )

        plt.close()

    # 3. PCA VISUALIZATION
    def plot_pca(self):

        print("Running PCA...")

        # SMALL SAMPLE FOR SPEED
        X_sample = self.X[:5000]
        y_sample = self.y[:5000]

        pca = PCA(n_components=2)

        X_pca = pca.fit_transform(X_sample)

        plt.figure(figsize=(10, 8))

        scatter = plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=y_sample,
            cmap='tab10',
            s=10
        )

        plt.colorbar(scatter)

        plt.title("PCA Feature Visualization")

        plt.savefig(
            OUTPUT_DIR / "pca_visualization.png"
        )

        plt.close()

    # 4. TSNE VISUALIZATION
    def plot_tsne(self):

        print("Running t-SNE...")

        # SMALL SAMPLE FOR SPEED
        X_sample = self.X[:2000]
        y_sample = self.y[:2000]

        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=30
        )

        X_tsne = tsne.fit_transform(X_sample)

        plt.figure(figsize=(10, 8))

        scatter = plt.scatter(
            X_tsne[:, 0],
            X_tsne[:, 1],
            c=y_sample,
            cmap='tab10',
            s=10
        )

        plt.colorbar(scatter)

        plt.title("t-SNE Feature Visualization")

        plt.savefig(
            OUTPUT_DIR / "tsne_visualization.png"
        )

        plt.close()

    # RUN ALL
    def analyze(self):

        self.plot_class_distribution()

        self.plot_feature_histogram()

        self.plot_pca()

        self.plot_tsne()

        print("Feature analysis completed")
