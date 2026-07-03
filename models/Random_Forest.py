import numpy as np
import config

from utils.sliding import sliding_window
from utils.sliding import vote_prediction
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)


def train_random_forest(
    train_set,
    val_set,
    param_grid
):

    x_train = np.array([
        sample["feature"]
        for sample in train_set
    ])

    y_train = np.array([
        sample["label"]
        for sample in train_set
    ])

    x_val = np.array([
        sample["feature"]
        for sample in val_set
    ])

    y_val = np.array([
        sample["label"]
        for sample in val_set
    ])

    best_model = None
    best_score = -1

    for params in param_grid:

        model = RandomForestClassifier(
            **params,
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            x_train,
            y_train
        )

        pred = model.predict(
            x_val
        )

        score = accuracy_score(
            y_val,
            pred
        )

        if score > best_score:

            best_score = score
            best_model = model

    return {
        "best_model": best_model,
        "best_score": best_score
    }


def evaluate_random_forest(model,test_set):

    y_true = []
    y_pred = []
    y_prob = []

    for sample in test_set:

        feature_sequence = sample["all_features"]
        true_label = sample["label"]

        # Sliding Window
        windows = sliding_window(
            feature_sequence,
            window_size=config.WINDOW_SIZE,
            stride=config.STRIDE
        )

        if len(windows) == 0:
            continue


        window_predictions = []
        window_probabilities = []

        for window in windows:

            x = window.reshape(1, -1)

            pred = model.predict(x)[0]
            prob = model.predict_proba(x)[0, 1]

            window_predictions.append(pred)
            window_probabilities.append(prob)


        final_prediction = vote_prediction(
            window_predictions,
            vote_size=config.VOTE_SIZE,
            threshold=config.VOTE_THRESHOLD
        )

        final_probability = np.max(window_probabilities)

        y_true.append(true_label)
        y_pred.append(final_prediction)
        y_prob.append(final_probability)

    # Evaluation
    acc = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    # ROC
    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob
    )
    
    roc_auc = auc(
        fpr,
        tpr
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc
    }