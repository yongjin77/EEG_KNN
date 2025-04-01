from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
from collections import Counter
from typing import Tuple, List, Any, Optional


class KNN:
    """
    K-Nearest Neighbors classifier implementation.

    This class implements the KNN algorithm for classification with
    Euclidean distance metric.
    """

    def __init__(self, k: int = 5):
        """
        Initialize the KNN classifier.

        Args:
            k: Number of neighbors to consider for classification
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the KNN model to the training data.

        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training labels of shape (n_samples,)
        """
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Calculate the Euclidean distance between two points.

        Args:
            x1: First point
            x2: Second point

        Returns:
            float: Euclidean distance between x1 and x2
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Args:
            X: Test samples of shape (n_samples, n_features)

        Returns:
            np.ndarray: Predicted class labels
        """
        # Vectorized implementation for better performance
        if len(X) > 1000:
            # For large datasets, process in batches to avoid memory issues
            batch_size = 1000
            predictions = []

            for i in range(0, len(X), batch_size):
                batch_X = X[i:i + batch_size]
                # Calculate distances between all points in the batch and training points
                distances = np.sqrt(((self.X_train[:, np.newaxis] - batch_X.T[np.newaxis, :]) ** 2).sum(axis=2))

                # Get indices of k nearest neighbors for each point
                k_indices = np.argpartition(distances, self.k, axis=0)[:self.k]

                # Get corresponding labels
                k_nearest_labels = self.y_train[k_indices]

                # Use majority voting to predict labels
                batch_predictions = [Counter(k_nearest_labels[:, i]).most_common(1)[0][0]
                                     for i in range(batch_X.shape[0])]
                predictions.extend(batch_predictions)

            return np.array(predictions)
        else:
            # For smaller datasets, use the original implementation
            return np.array([self._predict(x) for x in X])

    def _predict(self, x: np.ndarray) -> Any:
        """
        Predict class label for a single sample.

        Args:
            x: Test sample

        Returns:
            Any: Predicted class label
        """
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


def apply_knn(X: np.ndarray, y: np.ndarray, test_size: float = 0.2,
              k: int = 5, random_state: int = 77) -> Tuple[KNN, float]:
    """
    Apply KNN classification to the dataset and evaluate performance.

    Args:
        X: Features of shape (n_samples, n_features)
        y: Labels of shape (n_samples,)
        test_size: Proportion of the dataset to include in the test split
        k: Number of neighbors
        random_state: Random seed for reproducibility

    Returns:
        Tuple[KNN, float]: Trained KNN model and accuracy
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    knn = KNN(k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"accuracy: {accuracy:.2f}")

    print("report:")
    print(classification_report(y_test, y_pred))

    return knn, accuracy