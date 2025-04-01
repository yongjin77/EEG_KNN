import os
import numpy as np
from collections import Counter
from typing import List, Tuple, Any
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from ProgressKNN import ProgressKNN
from multi_case_knn_script import collect_all_data


def bootstrap_sampling(X: np.ndarray, y: np.ndarray, case_ids: List[str],
                       n_samples: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Perform bootstrap sampling on the input data"""
    indices = np.random.choice(len(X), n_samples, replace=True)
    return X[indices], y[indices], [case_ids[i] for i in indices]


def train_knn_model(X: np.ndarray, y: np.ndarray, case_ids: List[str],
                    k: int) -> Any:
    """Train a single KNN model"""
    knn = ProgressKNN(k)
    knn.fit(X, y, case_ids)
    return knn


def parallel_ensemble_knn(X_train: np.ndarray, y_train: np.ndarray,
                          case_ids_train: List[str], X_test: np.ndarray,
                          case_ids_test: List[str], y_test: np.ndarray,
                          k: int = 5, n_estimators: int = 5,
                          sample_ratio: float = 0.8) -> Tuple[float, List[Any], np.ndarray]:
    """
    Parallel implementation of ensemble KNN using joblib
    Returns accuracy, trained models, and predictions
    """
    n_samples = int(sample_ratio * len(X_train))

    # Generate bootstrap datasets
    bootstrap_datasets = [
        bootstrap_sampling(X_train, y_train, case_ids_train, n_samples)
        for _ in range(n_estimators)
    ]

    # Train models in parallel
    n_jobs = min(n_estimators, os.cpu_count())
    models = Parallel(n_jobs=n_jobs)(
        delayed(train_knn_model)(X, y, case_ids, k)
        for X, y, case_ids in bootstrap_datasets
    )

    # Get predictions from all models
    predictions = np.array([
        model.predict(X_test, case_ids_test) for model in models
    ])

    # Majority voting
    final_predictions = np.array([
        Counter(predictions[:, i]).most_common(1)[0][0]
        for i in range(len(X_test))
    ])

    accuracy = accuracy_score(y_test, final_predictions)
    return accuracy, models, final_predictions


if __name__ == "__main__":
    # Configuration
    DATA_DIR = "/Users/yongjindu/Desktop/EEG/sleep-edf-database-expanded-1.0.0/sleep-cassette/"
    TRAIN_RATIO = 0.8
    K = 5
    N_ESTIMATORS = 5
    RANDOM_STATE = 77

    # Load and prepare data
    print("Loading EEG data...")
    X_all, y_all, case_ids_all = collect_all_data(DATA_DIR)
    print(f"Total samples: {len(X_all)}")

    # Split dataset
    X_train, X_test, y_train, y_test, case_ids_train, case_ids_test = \
        train_test_split(X_all, y_all, case_ids_all,
                         test_size=1 - TRAIN_RATIO,
                         random_state=RANDOM_STATE)

    # Train and evaluate model
    print("Training ensemble KNN models in parallel...")
    accuracy, models, predictions = parallel_ensemble_knn(
        X_train, y_train, case_ids_train,
        X_test, case_ids_test, y_test,
        k=K, n_estimators=N_ESTIMATORS
    )

    # Output results
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    print(f"\nOverall Accuracy: {accuracy:.4f}")