import config
from preprocessing.reader import label_reader
from preprocessing.reader import sensor_reader
from preprocessing.dataset import dataset_builder
from preprocessing.window import build_windows
from preprocessing.feature import feature_extractor

from utils.split import split_loso
from utils.result import save_result
from utils.run import run_model

from models.Random_Forest import (
    train_random_forest,
    evaluate_random_forest
)

from models.SVM import (
    train_svm,
    evaluate_svm
)

from models.Transformer import (
    train_transformer,
    evaluate_transformer
)


def main():

    print("Reading Dataset...")

    # Read Data
    label_dict = label_reader()
    sensor_dict = sensor_reader()

    # Merge Dataset
    print("Building Dataset...")

    dataset = dataset_builder()

    # Window
    print("Generating Window Dataset...")
    window_dataset = build_windows(dataset)

    # Feature Extraction
    print("Generating Feature Dataset...")
    feature_dataset = feature_extractor(window_dataset)


    # LOSO Split
    feature_loso = split_loso(feature_dataset)
    sequence_loso = split_loso(window_dataset)


    run_model(
        "Random Forest",
        train_random_forest,
        evaluate_random_forest,
        feature_loso,
        save_result,
        param_grid=config.RF_PARAM_GRID
    )

    run_model(
        "SVM",
        train_svm,
        evaluate_svm,
        feature_loso,
        save_result,
        param_grid=config.SVM_PARAM_GRID
    )

    run_model(
        "Transformer",
        train_transformer,
        evaluate_transformer,
        sequence_loso,
        save_result,
        param_grid=config.TF_PARAM_GRID
    )


    print("\n")
    print("=" * 70)
    print("Experiment Finished.")
    print("=" * 70)


if __name__ == "__main__":
    main()