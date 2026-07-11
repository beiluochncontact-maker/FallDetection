import matplotlib
import config
from preprocessing.reader import label_reader
from preprocessing.reader import sensor_reader
from preprocessing.dataset import dataset_builder

from utils.split import split_loso
from utils.result import save_result
from utils.run import run_model
from utils.device import get_device
from utils.feature_cache import prepare_feature_datasets

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

    get_device(verbose=True)

    print("Reading Dataset...")

    # Read Data
    #label_dict = label_reader()
    #sensor_dict = sensor_reader()
    #print("Label Dict:", label_dict)
    #print("Sensor Dict:", sensor_dict)

    # Merge Dataset
    print("Building Dataset...")

    dataset = dataset_builder()

    matplotlib.use("Agg")
    visualizer = DataVisualizer()
    visualizer.visualize_all(dataset)

    # Feature Dataset (cached under output/Feature Dataset)
    train_features, test_features = prepare_feature_datasets(dataset)

    # LOSO Split
    loso_dataset = split_loso(train_features, test_features)

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

    '''
    run_model(
        "Transformer",
        train_transformer,
        evaluate_transformer,
        loso_dataset,
        save_result,
        param_grid=config.TF_PARAM_GRID
    )
    '''
    
    for model in [
        "Random Forest",
        "SVM",
        #"Transformer"
    ]:
        ResultVisualizer(model).visualize_all()

    print("\n")
    print("=" * 70)
    print("Experiment Finished.")
    print("=" * 70)


if __name__ == "__main__":
    main()
