import matplotlib
matplotlib.use("Agg")
import config
from preprocessing.reader import label_reader
from preprocessing.reader import sensor_reader
from preprocessing.dataset import dataset_builder
from preprocessing.window import build_windows
from preprocessing.feature import feature_extractor

from utils.split import split_loso
from utils.result import save_result
from utils.run import run_model

from visualization.data_visualizer import DataVisualizer
from visualization.result_visualizer import ResultVisualizer
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
    #label_dict = label_reader()
    #sensor_dict = sensor_reader()
    #print("Label Dict:", label_dict)
    #print("Sensor Dict:", sensor_dict)

    # Merge Dataset
    print("Building Dataset...")

    dataset = dataset_builder()

    visualizer = DataVisualizer()
    visualizer.visualize_all(dataset)

    # Window
    print("Generating Window Dataset...")
    window_dataset = build_windows(dataset)

    # Feature Extraction
    print("Generating Feature Dataset...")
    dataset = feature_extractor(window_dataset)
    

    # LOSO Split
    loso_dataset = split_loso(dataset)

    print("Training Models...")
    run_model(
        "Random Forest",
        train_random_forest,
        evaluate_random_forest,
        loso_dataset,
        save_result,
        param_grid=config.RF_PARAM_GRID
    )

    run_model(
        "SVM",
        train_svm,
        evaluate_svm,
        loso_dataset,
        save_result,
        param_grid=config.SVM_PARAM_GRID
    )

    run_model(
        "Transformer",
        train_transformer,
        evaluate_transformer,
        loso_dataset,
        save_result,
        param_grid=config.TF_PARAM_GRID
    )

    for model in [
        "Random Forest",
        "SVM",
        "Transformer"
    ]:
        ResultVisualizer(model).visualize_all()

    print("\n")
    print("=" * 70)
    print("Experiment Finished.")
    print("=" * 70)


if __name__ == "__main__":
    main()