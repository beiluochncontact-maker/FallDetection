from pathlib import Path
import joblib
import config
from preprocessing.window import build_windows
from preprocessing.feature import feature_extractor


TRAIN_CACHE_NAME = "train_features.joblib"
TEST_CACHE_NAME = "test_features.joblib"


def _cache_dir():
    path = Path(config.FEATURE_DATASET_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _is_usable(path):
    return path.exists() and path.stat().st_size > 0


def _save(features, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(features, tmp, compress=3)
    tmp.replace(path)
    print(f"[Saved] {path} ({len(features)} samples)")


def _load(path):
    features = joblib.load(path)
    print(f"[Loaded] {path} ({len(features)} samples)")
    return features


def prepare_feature_datasets(dataset, rebuild=None):
    """
    Load cached train/test features from output/Feature Dataset,
    or build, save, and return them.
    """
    if rebuild is None:
        rebuild = getattr(config, "REBUILD_FEATURES", False)

    cache_dir = _cache_dir()
    train_path = cache_dir / TRAIN_CACHE_NAME
    test_path = cache_dir / TEST_CACHE_NAME

    if (
        _is_usable(train_path)
        and _is_usable(test_path)
        and not rebuild
    ):
        print("Loading Feature Dataset...")
        try:
            return _load(train_path), _load(test_path)
        except Exception as e:
            print(f"Cache load failed ({type(e).__name__}), rebuilding...")
            for path in (train_path, test_path):
                path.unlink(missing_ok=True)

    print("Generating Train Window Dataset...")
    train_windows = build_windows(dataset, mode="train")

    print("Generating Test Window Dataset...")
    test_windows = build_windows(dataset, mode="test")

    print("Generating Train Feature Dataset...")
    train_features = feature_extractor(train_windows)

    print("Generating Test Feature Dataset...")
    test_features = feature_extractor(test_windows)

    print("Saving Feature Dataset...")
    _save(train_features, train_path)
    _save(test_features, test_path)

    return train_features, test_features
