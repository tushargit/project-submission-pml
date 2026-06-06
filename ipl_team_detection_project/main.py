import argparse
import numpy as np
from src.dataset_builder import DatasetBuilder
from src.train import TeamClassifierTrainer
from src.inference import IPLInference

from src.config import CSV_PATH
from src.config import MODEL_PATH
from src.utils import load_image, resize_image
from src.visualization import draw_predictions,draw_cell_numbers
from src.feature_analysis import FeatureAnalyzer

import cv2


def build_dataset():

    builder = DatasetBuilder(CSV_PATH)

    X, y = builder.build_dataset()
    builder.save_features(X, y)
    print(f"Dataset Built")
    print(f"X Shape: {X.shape}")
    print(f"y Shape: {y.shape}")

def train_model():

    data = np.load("features.npz")

    X = data["X"]
    y = data["y"]

    print("Features Loaded")

    print(X.shape)
    print(y.shape)

    # TEMP FAST TRAINING
    #X = X[:100000]
    #y = y[:100000]

    print("Reduced Dataset Shape")

    print(X.shape)
    print(y.shape)

    trainer = TeamClassifierTrainer()

    trainer.train(X, y, use_smote=True)

def evaluate_model():

    print("Evaluation placeholder")


def run_inference(image_path):

    inference = IPLInference(MODEL_PATH)

    predictions = inference.predict_image(image_path)

    print("Predictions:")
    print(predictions)
    predictions = inference.predict_image(image_path)

    print("Predictions:")
    print(predictions)

    inference.save_predictions(
        image_path=image_path,
        predictions=predictions,
        save_path="outputs/predictions.csv"
    )

    print("CSV saved to outputs/predictions.csv")
    image = load_image(image_path)

    visualized = draw_predictions(image, predictions)

    cv2.imwrite(
        "outputs/prediction_overlay_new.jpg",
        visualized
    )

    print("Visualization saved")
    img = load_image(image_path)
    img = resize_image(img)

    numbered = draw_cell_numbers(img)

    cv2.imwrite(
        "cell_numbers.png",
        numbered
    )
def analyze_features():

    data = np.load("features.npz")

    X = data["X"]
    y = data["y"]

    analyzer = FeatureAnalyzer(X, y)

    analyzer.analyze()

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "build",
            "train",
            "evaluate",
            "infer",
            "analyze"
        ]

    )

    parser.add_argument(
        "--image_path",
        type=str,
        default=None
    )

    args = parser.parse_args()

    if args.mode == "build":
        build_dataset()

    elif args.mode == "train":
        train_model()

    elif args.mode == "evaluate":
        evaluate_model()

    elif args.mode == "infer":

        if args.image_path is None:
            raise ValueError("Provide --image_path")

        run_inference(args.image_path)

    elif args.mode == "analyze":
        analyze_features()
    
if __name__ == "__main__":
    main()