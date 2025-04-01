import KNN
from collections import Counter
import numpy as np

class ProgressKNN(KNN.KNN):
    def __init__(self, k):
        super().__init__(k)
        self.X_train = None
        self.y_train = None
        self.case_ids_train = None

    def fit(self, X, y, case_ids):
        print("\nTraining")
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.case_ids_train = np.array(case_ids)
        print("training completed.")

    def predict(self, X, case_ids):
        print("\nPredicting.")
        y_pred = []
        total_samples = len(X)

        batch_size = 100
        last_reported_case = None
        for i in range(0, total_samples, batch_size):
            batch_end = min(i + batch_size, total_samples)
            batch_X = X[i:batch_end]
            batch_case_ids = case_ids[i:batch_end]

            # Vectorized distance calculation
            distances = np.sqrt(((self.X_train[:, np.newaxis] - batch_X) ** 2).sum(axis=2))

            k_indices = np.argpartition(distances, self.k, axis=0)[:self.k]

            k_nearest_labels = self.y_train[k_indices]
            batch_pred = []
            for labels in k_nearest_labels.T:
                most_common = Counter(labels).most_common(1)[0][0]
                batch_pred.append(most_common)

            y_pred.extend(batch_pred)

            # Report progress for each unique case in the batch
            for case_id in np.unique(batch_case_ids):
                if case_id != last_reported_case:
                    print(f"Prediction progress: {batch_end / total_samples * 100:.2f}% complete")
                    print(f"Current case: {case_id}")
                    last_reported_case = case_id

        print("prediction completed")
        return np.array(y_pred)

