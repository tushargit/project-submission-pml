import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score
)

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from collections import Counter

from src.config import MODEL_PATH, OUTPUT_DIR, CLASS_NAMES
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier
)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from xgboost import plot_importance

class TeamClassifierTrainer:

    def __init__(self):
        self.models = {

            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),

            'ExtraTrees': ExtraTreesClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),

            'XGBoost': XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                num_class=len(CLASS_NAMES),
                colsample_bytree=0.8,
                objective='multi:softprob',
                random_state=42,
                tree_method='hist',
                eval_metric='mlogloss',
                verbosity=0,
                n_jobs=-1
            ),

            'LogisticRegression': LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                n_jobs=-1
            )
        }
    def plot_class_distribution_after_sampling(self, y):

        counter = Counter(y)

        plt.figure(figsize=(12, 6))

        sns.barplot(
            x=list(counter.keys()),
            y=list(counter.values())
        )

        plt.title("Class Distribution After Sampling")

        plt.savefig(
            OUTPUT_DIR /
            "class_distribution_after_sampling.png"
        )

        plt.close()
    def plot_class_distribution(self, y):

        counter = Counter(y)

        plt.figure(figsize=(12, 6))

        sns.barplot(
            x=list(counter.keys()),
            y=list(counter.values())
        )

        plt.title("Class Distribution")

        plt.savefig(OUTPUT_DIR / 'class_distribution.png')

        plt.close()

    def train(self, X, y,use_smote):

        print("\n========== DATASET ==========\n")

        print("X Shape:", X.shape)
        print("y Shape:", y.shape)
        X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    stratify=y,
                    random_state=42
        )
        self.plot_class_distribution(y)
        print("\nOriginal Distribution")

        print(Counter(y))
        team_mask = y_train != 0

        team_X = X_train[team_mask]
        team_y = y_train[team_mask]

        no_team_X = X_train[y_train == 0]
        no_team_y = y_train[y_train == 0]

        # Keep same number of No Team samples as Team samples

        team_counts = Counter(team_y)

        largest_team = max(team_counts.values())

        print("Largest Team Class:", largest_team)

        keep_idx = np.random.choice(
            len(no_team_X),
            size=largest_team,
            replace=False
        )

        X_train = np.vstack([
            team_X,
            no_team_X[keep_idx]
        ])

        y_train = np.concatenate([
            team_y,
            no_team_y[keep_idx]
        ])

        print("\nAFTER SAMPLING")

        print("X_train Shape:", X_train.shape)
        print("y_train Shape:", y_train.shape)
        print("\nTrain Distribution After Sampling")

        print(Counter(y_train))
        self.plot_class_distribution_after_sampling(y_train)
        

        best_model = None
        best_score = 0
        results = []
        for name, model in self.models.items():

            print(f"\n========== TRAINING STARTED: {name} ==========\n")

            model.fit(X_train, y_train)

            print(f"\n========== TRAINING COMPLETED: {name} ==========\n")

            predictions = model.predict(X_test)

            print(f"\n========== PREDICTION COMPLETED: {name} ==========\n")
            # Generate confusion matrix
            cm = confusion_matrix(y_test, predictions)

            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=list(CLASS_NAMES.values())
            )

            fig, ax = plt.subplots(figsize=(10, 10))

            disp.plot(
                ax=ax,
                cmap="Blues",
                colorbar=False
            )

            plt.title(f"{name} Confusion Matrix")

            plt.savefig(
                OUTPUT_DIR / f"{name}_confusion_matrix.png",
                bbox_inches="tight"
            )

            plt.close()
            #end confusion_matrix
            accuracy = accuracy_score(y_test, predictions)

            f1 = f1_score(
                y_test,
                predictions,
                average='weighted'
            )

            print(f"{name} Accuracy: {accuracy:.4f}")
            print(f"{name} F1 Score: {f1:.4f}")

            report = classification_report(
                y_test,
                predictions,
                zero_division=0
            )

            print(report)

            with open(
                OUTPUT_DIR / f'{name}_report.txt',
                'w'
            ) as f:

                f.write(report)

            if f1 > best_score:
                if name == "XGBoost":
                    plt.figure(figsize=(12,8))

                    plot_importance(
                        model,
                        max_num_features=20
                    )

                    plt.tight_layout()

                    plt.savefig(
                        OUTPUT_DIR /
                        "xgboost_feature_importance.png"
                    )

                    plt.close()

                best_score = f1
                best_model = model

                with open(MODEL_PATH, 'wb') as f:

                    pickle.dump(best_model, f)

                print(f"\nBEST MODEL SAVED: {name}\n")
            results.append({
                "model": name,
                "accuracy": accuracy,
                "f1": f1
            })
        models = [r["model"] for r in results]
        f1_scores = [r["f1"] for r in results]

        plt.figure(figsize=(8,5))

        plt.bar(
            models,
            f1_scores
        )

        plt.title("Model Comparison (Weighted F1)")
        plt.ylabel("F1 Score")

        plt.savefig(
            OUTPUT_DIR / "model_comparison_f1.png"
        )

        plt.close()
        return best_model
