import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from src.config import MODEL_PATH


class Evaluator:

    def __init__(self):

        with open(MODEL_PATH, 'rb') as f:
            self.model = pickle.load(f)

    def evaluate(self, X_test, y_test, output_dir):

        predictions = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        print(f"Accuracy: {accuracy:.4f}")

        report = classification_report(y_test, predictions)

        with open(output_dir / 'classification_report.txt', 'w') as f:
            f.write(report)

        cm = confusion_matrix(y_test, predictions)

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d')

        plt.title("Confusion Matrix")

        plt.savefig(output_dir / 'confusion_matrix.png')
        plt.close()