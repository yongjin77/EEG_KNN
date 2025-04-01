import os
import numpy as np
from collections import Counter
from typing import List, Tuple, Any, Optional, Dict
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from ProgressKNN import ProgressKNN
from multi_case_knn_script import collect_all_data
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def bootstrap_sampling(X: np.ndarray, y: np.ndarray, case_ids: List[str],
                       n_samples: int = None,
                       random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Perform bootstrap sampling on the input data.

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Labels of shape (n_samples,)
        case_ids: List of case IDs
        n_samples: Number of samples to draw (default: len(X))
        random_state: Random seed for reproducibility

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]: Bootstrapped features, labels, and case IDs
    """
    if n_samples is None:
        n_samples = len(X)

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Generate bootstrap indices
    indices = np.random.choice(len(X), n_samples, replace=True)

    # Return bootstrapped data
    return X[indices], y[indices], [case_ids[i] for i in indices]


def train_knn_model(X: np.ndarray, y: np.ndarray, case_ids: List[str],
                    k: int) -> ProgressKNN:
    """
    Train a single KNN model.

    Args:
        X: Feature matrix
        y: Labels
        case_ids: List of case IDs
        k: Number of neighbors

    Returns:
        ProgressKNN: Trained KNN model
    """
    try:
        knn = ProgressKNN(k)
        knn.fit(X, y, case_ids)
        return knn
    except Exception as e:
        logger.error(f"Error training KNN model: {str(e)}")
        return None


def parallel_ensemble_knn(X_train: np.ndarray, y_train: np.ndarray,
                          case_ids_train: List[str], X_test: np.ndarray,
                          case_ids_test: List[str], y_test: np.ndarray,
                          k: int = 5, n_estimators: int = 5,
                          sample_ratio: float = 0.8,
                          random_state: Optional[int] = None) -> Tuple[float, List[Any], np.ndarray]:
    """
    Parallel implementation of ensemble KNN using joblib.

    Args:
        X_train: Training features
        y_train: Training labels
        case_ids_train: Training case IDs
        X_test: Test features
        case_ids_test: Test case IDs
        y_test: Test labels
        k: Number of neighbors
        n_estimators: Number of KNN models in the ensemble
        sample_ratio: Ratio of samples to use for each bootstrap
        random_state: Random seed for reproducibility

    Returns:
        Tuple[float, List[Any], np.ndarray]: Accuracy, trained models, and predictions
    """
    n_samples = int(sample_ratio * len(X_train))

    logger.info(f"Creating ensemble with {n_estimators} KNN models (k={k})")
    logger.info(f"Bootstrap sample size: {n_samples} ({sample_ratio:.2f} of training data)")

    # Set random seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)

    # Generate bootstrap datasets with different random states for each estimator
    bootstrap_datasets = [
        bootstrap_sampling(X_train, y_train, case_ids_train, n_samples,
                           random_state=random_state + i if random_state else None)
        for i in range(n_estimators)
    ]

    # Train models in parallel
    n_jobs = min(n_estimators, os.cpu_count())
    logger.info(f"Training models in parallel using {n_jobs} cores")

    models = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(train_knn_model)(X, y, case_ids, k)
        for X, y, case_ids in bootstrap_datasets
    )

    # Filter out any failed models
    models = [model for model in models if model is not None]

    if not models:
        logger.error("All models failed to train")
        return 0.0, [], np.array([])

    logger.info(f"Successfully trained {len(models)} models")

    # Get predictions from all models
    logger.info("Generating predictions from all models")
    predictions = np.array([
        model.predict(X_test, case_ids_test) for model in models
    ])

    # Majority voting
    logger.info("Performing majority voting")
    final_predictions = np.array([
        Counter(predictions[:, i]).most_common(1)[0][0]
        for i in range(len(X_test))
    ])

    accuracy = accuracy_score(y_test, final_predictions)
    logger.info(f"Ensemble accuracy: {accuracy:.4f}")

    return accuracy, models, final_predictions


def main(data_dir: str, train_ratio: float = 0.8, k: int = 5,
         n_estimators: int = 5, random_state: int = 77):
    """
    Main function to run the ensemble KNN classification.

    Args:
        data_dir: Directory containing the data
        train_ratio: Ratio of data to use for training
        k: Number of neighbors
        n_estimators: Number of KNN models in the ensemble
        random_state: Random seed for reproducibility
    """
    # Load and prepare data
    logger.info(f"Loading EEG data from {data_dir}")
    X_all, y_all, case_ids_all = collect_all_data(data_dir)
    logger.info(f"Total samples: {len(X_all)}")

    # Split dataset
    logger.info(f"Splitting dataset (train_ratio={train_ratio}, random_state={random_state})")
    X_train, X_test, y_train, y_test, case_ids_train, case_ids_test = \
        train_test_split(X_all, y_all, case_ids_all,
                         test_size=1 - train_ratio,
                         random_state=random_state)

    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")

    # Train and evaluate model
    logger.info("Training ensemble KNN models in parallel...")
    accuracy, models, predictions = parallel_ensemble_knn(
        X_train, y_train, case_ids_train,
        X_test, case_ids_test, y_test,
        k=k, n_estimators=n_estimators, random_state=random_state
    )

    # Output results
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, predictions))
    logger.info(f"\nOverall Accuracy: {accuracy:.4f}")

    return accuracy, models, predictions


if __name__ == "__main__":
    # Configuration (should ideally be loaded from a config file)
    config = {
        "DATA_DIR": "/Users/yongjindu/Desktop/EEG/sleep-edf-database-expanded-1.0.0/sleep-cassette/",
        "TRAIN_RATIO": 0.8,
        "K": 5,
        "N_ESTIMATORS": 5,
        "RANDOM_STATE": 77
    }

    main(
        data_dir=config["DATA_DIR"],
        train_ratio=config["TRAIN_RATIO"],
        k=config["K"],
        n_estimators=config["N_ESTIMATORS"],
        random_state=config["RANDOM_STATE"]
    )