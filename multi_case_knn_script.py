import os
import numpy as np
import ProcessEEG
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from ProgressKNN import ProgressKNN
import pickle

def collect_all_data(directory_path, cache_file='processed_data_cache.pkl'):
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
    X_train, X_test, y_train, y_test, case_ids_train, case_ids_test = train_test_split(
        X, y, case_ids, test_size=test_size, random_state=77
    )
    knn = ProgressKNN(k)
    knn.fit(X_train, y_train, case_ids_train)
    y_pred = knn.predict(X_test, case_ids_test)
    accuracy = accuracy_score(y_test, y_pred)
    return knn, accuracy

if __name__ == "__main__":
    directory_path = "/Users/yongjindu/Desktop/EEG/sleep-edf-database-expanded-1.0.0/sleep-cassette/"

    start_time = time.time()
    X_all, y_all, case_ids_all = collect_all_data(directory_path)

    print(f"\nTotal number of samples collected: {len(X_all)}")


    print("\nApplying KNN to the entire dataset...")
    k = 3
    test_size = 0.2
    knn, accuracy = apply_knn_with_progress(X_all, y_all, case_ids_all, test_size, k)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n Results:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Total processing time: {total_time:.2f} seconds")