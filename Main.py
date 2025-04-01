import ProcessEEG
import KNN
if __name__ == "__main__":
    # example for 1
    eeg_file = "SC4001E0"
    anno_file = "SC4001EC"

    X, y = ProcessEEG.prepare_data_for_knn(eeg_file, anno_file)

    knn, accuracy = KNN.apply_knn(X, y, 0.2, 3)

