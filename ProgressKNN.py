import KNN
from collections import Counter
import numpy as np
import time
from typing import List, Optional, Any, Tuple


class ProgressKNN(KNN.KNN):
    """
    Enhanced KNN classifier with progress reporting.

    This class extends the basic KNN implementation to provide
    progress reporting during training and prediction.
    """

    def __init__(self, k: int = 5):
        """
        Initialize the ProgressKNN classifier.

        Args:
            k: Number of neighbors to consider
        """
        super().__init__(k)
        self.X_train = None
        self.y_train = None
        self.case_ids_train = None

    def fit(self, X: np.ndarray, y: np.ndarray, case_ids: List[str]) -> None:
        """
        Fit the KNN model to the training data with progress reporting.

        Args:
            X: Training features
            y: Training labels
            case_ids: Case IDs for each sample
        """
        print("\nTraining KNN model...")
        start_time = time.time()

        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.case_ids_train = np.array(case_ids)

        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds.")
        print(f"Model trained on {len(X)} samples with {len(set(case_ids))} unique cases.")

    def predict(self, X: np.ndarray, case_ids: List[str]) -> np.ndarray:
        """
        Predict class labels for samples in X with progress reporting.

        Uses vectorized operations for better performance and reports progress.

        Args:
            X: Test features
            case_ids: Case IDs for each sample

        Returns:
            np.ndarray: Predicted class labels
        """
        print("\nPredicting...")
        start_time = time.time()

        y_pred = []
        total_samples = len(X)

        # Use larger batch size for faster processing
        batch_size = min(1000, total_samples)

        # Track progress
        last_reported_percentage = -1
        last_reported_case = None

        for i in range(0, total_samples, batch_size):
            batch_end = min(i + batch_size, total_samples)
            batch_X = X[i:batch_end]
            batch_case_ids = case_ids[i:batch_end]

            # Calculate current progress percentage
            current_percentage = int((i / total_samples) * 100)

            # Report progress if it's changed significantly
            if current_percentage >= last_reported_percentage + 10:
                print(f"Prediction progress: {current_percentage}% complete")
                last_reported_percentage = current_percentage

            # Report current case being processed
            unique_batch_cases = set(batch_case_ids)
            for case_id in unique_batch_cases:
                if case_id != last_reported_case:
                    print(f"Processing case: {case_id}")
                    last_reported_case = case_id

            # Vectorized distance calculation - optimized for memory usage
            # This approach is more memory-efficient for large datasets
            batch_pred = []
            for j, sample in enumerate(batch_X):
                # Calculate distances for this sample
                distances = np.sqrt(np.sum((self.X_train - sample) ** 2, axis=1))

                # Get indices of k nearest neighbors
                k_indices = np.argpartition(distances, self.k)[:self.k]

                # Get the corresponding labels
                k_nearest_labels = self.y_train[k_indices]

                # Use majority voting
                most_common = Counter(k_nearest_labels).most_common(1)[0][0]
                batch_pred.append(most_common)

            y_pred.extend(batch_pred)

        end_time = time.time()
        print(f"Prediction completed in {end_time - start_time:.2f} seconds.")

        return np.array(y_pred)

    def predict_efficient(self, X: np.ndarray, case_ids: List[str]) -> np.ndarray:
        """
        Alternative prediction method using full vectorization.

        This method is faster for smaller datasets but may use more memory.

        Args:
            X: Test features
            case_ids: Case IDs for each sample

        Returns:
            np.ndarray: Predicted class labels
        """
        print("\nPredicting (efficient method)...")
        start_time = time.time()

        total_samples = len(X)

        # Use batching for larger datasets to avoid memory issues
        if total_samples > 5000:
            return self.predict(X, case_ids)

        # Full vectorized implementation for smaller datasets
        # Calculate distances between all training and test points at once
        # Shape: (n_train, n_test)
        distances = np.sqrt(((self.X_train[:, np.newaxis] - X.T[np.newaxis, :]) ** 2).sum(axis=2))

        # Get indices of k nearest neighbors for each test point
        k_indices = np.argpartition(distances, self.k, axis=0)[:self.k]

        # Get the corresponding labels
        k_nearest_labels = self.y_train[k_indices]

        # Use majority voting for each test point
        y_pred = []
        for i in range(total_samples):
            labels = k_nearest_labels[:, i]
            most_common = Counter(labels).most_common(1)[0][0]
            y_pred.append(most_common)

        end_time = time.time()
        print(f"Prediction completed in {end_time - start_time:.2f} seconds.")

        return np.array(y_pred)
