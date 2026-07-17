from pathlib import Path

# Directories
ROOT = Path(__file__).resolve().parent
SENSOR_DATA_DIR = ROOT / "preprocessing" / "data" / "sensor_data"
LABEL_DATA_DIR = ROOT / "preprocessing" / "data" / "label_data"
OUTPUT_DIR = ROOT / "output"
FEATURE_DATASET_DIR = OUTPUT_DIR / "Feature Dataset"

REBUILD_FEATURES = False

SAMPLING_RATE = 100

# Window parameters
WINDOW_SIZE = 32
ACCEPT_RATE = 0.5
TRAIN_STRIDE = 32
TEST_STRIDE = 64
NON_FALL_MAX_WINDOWS = 8

# Soft probability aggregation
PROB_SMOOTH_SIZE = 2
PROB_THRESHOLD = 0.46

# Score aggregation for k>=2:
SCORE_AGG = "mean"
SCORE_ALPHA = 0.0
# alpha-k offline sweep
SCORE_ALPHA_RANGE = (-1.0, 1.0)
ALPHA_STRIDE = 0.25
SCORE_K = (2, 3, 4)

# Hard Negative Mining
ENABLE_HNM = True
HNM_HARD_FRACTION = 0.7  # fraction of negatives taken from hardest pool
HNM_ADL_BONUS = 0.15     # added to pilot P(fall) for known hard ADL tasks
HNM_HARD_ADL_TASKS = (2, 5, 10, 15, 16, 18, 19, 35, 36)


# Label parameters
LABEL_FALL = 1
LABEL_NON_FALL = 0
DESCRIPTION_MAP = {
    "Forward fall when trying to sit down": 0,
    "Backward fall when trying to sit down": 1,
    "lateral fall when trying to sit down": 2,
    "Forward fall when trying to get up": 3,
    "lateral fall when trying to get up": 4,
    "Forward fall while sitting, caused by fainting": 5,
    "lateral fall while sitting, caused by fainting": 6,
    "Backward fall while sitting, caused by fainting": 7,
    "Vertical(forward) fall while walking caused by fainting": 8,
    "Fall while walking, with use of hands in a table to dampen fall, caused by fainting": 9,
    "Forward fall while walking caused by a trip": 10,
    "Forward fall while jogging caused by a trip": 11,
    "Forward fall while walking caused by a slip": 12,
    "Forward lateral fall while walking caused by a slip": 13,
    "Backward fall while walking caused by a slip": 14
}

# Feature
FEATURE_COLUMNS = [
    "AccX", "AccY", "AccZ",
    "GyrX", "GyrY", "GyrZ",
    "EulerX", "EulerY", "EulerZ"
]

FEATURE_LIST = [
    "mean",
    "std",
    "max",
    "min",
    "range",
    "peak_to_peak",
    "rms",
    "variance",
    "median",
    "iqr",
    "energy",
    "skewness",
    "kurtosis",

    "acc_mag",
    "gyr_mag",
    "euler_mag",
    "acc_sma",
    "gyr_sma"
]

# Random Forest parameters
RF_PARAM_GRID = [
    {
        "n_estimators": 100,
        "max_depth": None
    }
]

# SVM parameters
SVM_PARAM_GRID = {
    "kernel": [
        "rbf"
    ],
    "C": [
        10,
        100
    ],
    "gamma": [
        "scale",
        0.01
    ]
}

# Transformer parameters
INPUT_DIM = 9
D_MODEL = 64
TF_PARAM_GRID = {
    "input_dim": 9,
    "window_size": WINDOW_SIZE,
    "num_classes": 2,
    "d_model": 64,
    "nhead": 8,
    "num_layers": 2,
    "dim_feedforward": 128,
    "dropout": 0.2,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "epochs": 30
}