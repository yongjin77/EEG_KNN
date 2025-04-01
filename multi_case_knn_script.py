import os
import numpy as np
import ProcessEEG
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from ProgressKNN import ProgressKNN
import pickle


def collect_all_data(directory_path, cache_file='processed_data_cache.pkl'):
    """
    Collect and prepare data from all case files in the directory.

    Args:
        directory_path: Path to the directory containing EEG files
        cache_file: Path to the cache file

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]: Features, labels, and case IDs
    """
    if os.path.exists(cache_file):
        print("Loading cached data...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    all_X = []
    all_y = []
    all_case_ids = []

    files = os.listdir(directory_path)

    case_files = {}
    for file in files:
        if file.endswith('-PSG.edf') or file.endswith('-Hypnogram.edf'):
            case_id = file[:6]
            if case_id not in case_files:
                case_files[case_id] = {'PSG': None, 'Hypnogram': None}

            if file.endswith('-PSG.edf'):
                case_files[case_id]['PSG'] = file[:-8]
            elif file.endswith('-Hypnogram.edf'):
                case_files[case_id]['Hypnogram'] = file[:-14]

    total_cases = len(case_files)
    processed_cases = 0

    for case_id, files in case_files.items():
        if files['PSG'] and files['Hypnogram']:
            processed_cases += 1
            print(f"\nProcessing case {processed_cases}/{total_cases}: {case_id}")

            try:
                X, y = ProcessEEG.prepare_data_for_knn(files['PSG'], files['Hypnogram'])
                all_X.append(X)
                all_y.append(y)
                all_case_ids.extend([case_id] * len(X))
                print(f"Case {case_id} data collected successfully.")
                print(f"Progress: {processed_cases / total_cases * 100:.2f}% complete")
            except Exception as e:
                print(f"Error collecting data from case {case_id}: {str(e)}")
        else:
            print(f"\nIncomplete files for case {case_id}. Skipping.")

    X_combined = np.vstack(all_X)
    y_combined = np.concatenate(all_y)

    # Cache the processed data
    print("Caching processed data...")
    with open(cache_file, 'wb') as f:
        pickle.dump((X_combined, y_combined, all_case_ids), f)

    return X_combined, y_combined, all_case_ids


def apply_knn_with_progress(X, y, case_ids, test_size, k):
    """
    Apply KNN with progress reporting and evaluate performance.

    Args:
        X: Features matrix
        y: Labels array
        case_ids: List of case IDs
        test_size: Proportion of data to use for testing
        k: Number of neighbors

    Returns:
        Tuple[ProgressKNN, float]: Trained model and accuracy
    """
    print(f"\nApplying KNN (k={k}) with test_size={test_size}")

    # Split data
    X_train, X_test, y_train, y_test, case_ids_train, case_ids_test = train_test_split(
        X, y, case_ids, test_size=test_size, random_state=77
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Train model
    start_time = time.time()
    knn = ProgressKNN(k)
    knn.fit(X_train, y_train, case_ids_train)

    # Generate predictions
    y_pred = knn.predict(X_test, case_ids_test)

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)

    end_time = time.time()
    training_time = end_time - start_time

    print(f"KNN training and prediction completed in {training_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f}")

    return knn, accuracy


if __name__ == "__main__":
    # Configuration
    directory_path = "/Users/yongjindu/Desktop/EEG/sleep-edf-database-expanded-1.0.0/sleep-cassette/"
    test_size = 0.2
    k = 3

    try:
        # Measure execution time
        start_time = time.time()

        # Load data
        print("Starting data collection process")
        X_all, y_all, case_ids_all = collect_all_data(directory_path)

        if len(X_all) == 0:
            print("No data available. Exiting.")
            exit()

        print(f"\nTotal number of samples collected: {len(X_all)}")
        print(f"Feature dimension: {X_all.shape[1]}")
        print(f"Unique labels: {np.unique(y_all)}")
        print(f"Unique cases: {len(set(case_ids_all))}")

        # Apply KNN
        print("\nApplying KNN to the entire dataset...")
        knn, accuracy = apply_knn_with_progress(X_all, y_all, case_ids_all, test_size, k)

        # Calculate execution time
        end_time = time.time()
        total_time = end_time - start_time

        # Output results
        print(f"\nResults Summary:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Total processing time: {total_time:.2f} seconds")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback

        print(traceback.format_exc())