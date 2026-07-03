import numpy as np
import config

from utils.sliding import sliding_window
from utils.sliding import vote_prediction
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def train_svm(
        train_set, 
        val_set, 
        param_grid
    ):

    x_train, y_train = [], []

    for sample in train_set:

        windows = sliding_window(
            sample["all_features"],
            window_size=32,
            stride=8
        )

        for w in windows:
            x_train.append(w.reshape(-1))
            y_train.append(sample["label"])

    x_train = np.array(x_train)
    y_train = np.array(y_train)


    # Validation set
    x_val, y_val = [], []

    for sample in val_set:

        windows = sliding_window(
            sample["all_features"],
            window_size=32,
            stride=8
        )

        for w in windows:
            x_val.append(w.reshape(-1))
            y_val.append(sample["label"])

    x_val = np.array(x_val)
    y_val = np.array(y_val)


    # Grid search
    best_model = None
    best_score = -1
    best_params = None

    for params in ParameterGrid(param_grid):

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                **params,
                class_weight="balanced",
                random_state=42
            ))
        ])

        model.fit(x_train, y_train)

        pred = model.predict(x_val)

        score = accuracy_score(y_val, pred)

        if score > best_score:
            best_score = score
            best_model = model
            best_params = params

    return {
        "best_model": best_model,
        "best_params": best_params,
        "best_score": best_score
    }


def evaluate_svm(
        model,
        test_set):

    y_true = []
    y_pred = []
    y_prob = []

    for sample in test_set:


        # Sliding window
        windows = sliding_window(
            sample["all_features"],
            window_size=config.WINDOW_SIZE,
            stride=config.STRIDE
        )

        if len(windows) == 0:
            continue

        window_preds = []
        window_probs = []

        for w in windows:

            x = w.reshape(1, -1)

            pred = model.predict(x)[0]

            # SVM probability
            SVC(probability=True)
            prob = model.predict_proba(x)[0, 1]

            window_preds.append(pred)
            window_probs.append(prob)


        # Window voting
        final_pred = vote_prediction(
            window_preds,
            vote_size=config.VOTE_SIZE,
            threshold=config.THRESHOLD
        )

        # sequence-level probability
        final_prob = np.mean(window_probs)

        y_true.append(sample["label"])
        y_pred.append(final_pred)
        y_prob.append(final_prob)


    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm
    }