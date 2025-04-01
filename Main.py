import os
import argparse
import logging
import time
import ProcessEEG
import KNN
from typing import Tuple, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments() -> Dict[str, Any]:
    """
    Parse command line arguments.

    Returns:
        Dict[str, Any]: Dictionary of arguments
    """
    parser = argparse.ArgumentParser(description='Sleep EEG Classification')

    parser.add_argument('--eeg-file', type=str, default="SC4001E0",
                        help='Base name of the EEG file')
    parser.add_argument('--anno-file', type=str, default="SC4001EC",
                        help='Base name of the annotation file')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to the data directory')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of the dataset to include in the test split')
    parser.add_argument('--k', type=int, default=3,
                        help='Number of neighbors for KNN')
    parser.add_argument('--random-state', type=int, default=77,
                        help='Random seed for reproducibility')

    return vars(parser.parse_args())


def process_single_case(eeg_file: str, anno_file: str, data_path: str = None,
                        test_size: float = 0.2, k: int = 3,
                        random_state: int = 77) -> Tuple[KNN.KNN, float]:
    """
    Process a single case and apply KNN classification.

    Args:
        eeg_file: Base name of the EEG file
        anno_file: Base name of the annotation file
        data_path: Path to the data directory
        test_size: Proportion of the dataset to include in the test split
        k: Number of neighbors
        random_state: Random seed for reproducibility

    Returns:
        Tuple[KNN.KNN, float]: Trained KNN model and accuracy
    """
    logger.info(f"Processing case: {eeg_file}")

    # Prepare data
    start_time = time.time()
    X, y = ProcessEEG.prepare_data_for_knn(eeg_file, anno_file, file_path=data_path)
    data_time = time.time() - start_time

    logger.info(f"Data preparation completed in {data_time:.2f} seconds")
    logger.info(f"Dataset shape: {X.shape}, {len(y)} labels")

    # Apply KNN
    logger.info(f"Applying KNN with k={k}, test_size={test_size}")
    knn, accuracy = KNN.apply_knn(X, y, test_size, k)

    return knn, accuracy


def main():
    """
    Main function to run the sleep EEG classification.
    """
    # Parse arguments
    args = parse_arguments()

    logger.info("Starting Sleep EEG Classification")
    logger.info(f"Parameters: {args}")

    # Process single case
    try:
        start_time = time.time()

        knn, accuracy = process_single_case(
            args['eeg_file'],
            args['anno_file'],
            args['data_path'],
            args['test_size'],
            args['k'],
            args['random_state']
        )

        total_time = time.time() - start_time

        logger.info(f"Results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Total execution time: {total_time:.2f} seconds")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()