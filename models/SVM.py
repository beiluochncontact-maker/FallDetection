import numpy as np

import config
from imblearn.under_sampling import RandomUnderSampler
from collections import defaultdict
from utils.hnm import flatten_train_arrays, mine_hard_negatives
from utils.sliding import trial_decision_from_probs
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)


def _make_svm_pipeline(params, probability: bool):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            **params,
            probability=probability,
            cache_size=1000,
            random_state=42,
        )),
    ])


def train_svm(
    train_set,
    val_set,
    param_grid
):

    x_train, y_train, trials = flatten_train_arrays(train_set)

    # Build Validation Data
    x_val = []
    y_val = []

    for subject_samples in val_set.values():

        for sample in subject_samples:

            x_val.append(sample["feature"])
            y_val.append(sample["label"])

    x_val = np.asarray(
        x_val,
        dtype=np.float32
    )

    y_val = np.asarray(
        y_val,
        dtype=np.int32
    )

    # Random Under Sampling for training data (pilot / baseline balance)
    rus = RandomUnderSampler(
        sampling_strategy=1.0,
        random_state=42
    )

    x_train_balance, y_train_balance = rus.fit_resample(
        x_train,
        y_train
    )

    print("*" * 3)
    print("Random Under Sampling")
    print("Before Sampling")

    unique, counts = np.unique(
        y_train,
        return_counts=True
    )

    for label, count in zip(unique, counts):
        print(f"Label {label}: {count}")

    print()

    print("After Sampling")

    unique, counts = np.unique(
        y_train_balance,
        return_counts=True
    )

    for label, count in zip(unique, counts):
        print(f"Label {label}: {count}")

    print("*" * 3)

    hnm_info = None
    if config.ENABLE_HNM:
        # Pilot model on ordinary RUS balance (fixed mild hyperparams)
        pilot_params = {"kernel": "rbf", "C": 10, "gamma": "scale"}
        print("HNM: fitting pilot SVM for hard-negative mining...")
        pilot = _make_svm_pipeline(pilot_params, probability=True)
        pilot.fit(x_train_balance, y_train_balance)
        x_train_balance, y_train_balance, hnm_info = mine_hard_negatives(
            x_train,
            y_train,
            trials,
            pilot,
        )
        print("HNM: rebuilt train balance")
        for key, value in hnm_info.items():
            print(f"  {key}: {value}")
        unique, counts = np.unique(y_train_balance, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  Label {label}: {count}")

    # Grid Search (no probability during search; refit best with probability)
    best_score = -1
    best_param = None
    seen_linear_c = set()

    for params in ParameterGrid(param_grid):

        # gamma is ignored for linear kernel; skip duplicate (C, linear) pairs
        if params.get("kernel") == "linear":
            c = params.get("C")
            if c in seen_linear_c:
                continue
            seen_linear_c.add(c)

        print(f"Training SVM : {params}")
        model = _make_svm_pipeline(params, probability=False)

        model.fit(
            x_train_balance,
            y_train_balance
        )

        pred = model.predict(
            x_val
        )

        score = accuracy_score(
            y_val,
            pred
        )

        print(f"Validation Accuracy : {score:.4f}")

        if score > best_score:

            best_score = score
            best_param = params

    best_model = _make_svm_pipeline(best_param, probability=True)
    best_model.fit(
        x_train_balance,
        y_train_balance
    )

    result = {
        "best_model": best_model,
        "best_score": best_score,
        "best_param": best_param,
    }
    if hnm_info is not None:
        result["hnm_info"] = hnm_info
    return result


def evaluate_svm(model, test_set):

    # Group windows by trial
    trial_dict = defaultdict(list)

    for sample in test_set:
        trial_key = (
            sample["subject"],
            sample["trial"]
        )
        trial_dict[trial_key].append(sample)

    # Evaluation Containers
    y_true = []
    y_pred = []
    y_prob = []

    # Evaluate Each Trial
    for (subject, trial), windows in trial_dict.items():
        # Sort windows by time
        windows.sort(
            key=lambda x: x["start"]
        )

        # Ground Truth
        # Trial中只要有一个Window属于Fall
        # 即认为该Trial为Fall
        trial_label = max(
            sample["label"]
            for sample in windows
        )

        # Soft window probabilities → trial score + aligned hard decision
        window_probabilities = []

        for sample in windows:
            feature = sample["feature"].reshape(1, -1)
            prob = model.predict_proba(feature)[0, 1]
            window_probabilities.append(prob)

        trial_prediction, trial_probability = trial_decision_from_probs(
            window_probabilities
        )

        y_true.append(trial_label)
        y_pred.append(trial_prediction)
        y_prob.append(trial_probability)

    # Metrics
    accuracy = accuracy_score(
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

    # ROC Curve
    if len(np.unique(y_true)) > 1:

        fpr, tpr, _ = roc_curve(
            y_true,
            y_prob
        )

        roc_auc = auc(
            fpr,
            tpr
        )

    else:
        fpr = np.array([0, 1])
        tpr = np.array([0, 1])
        roc_auc = 0.0

    # Print Results
    print("+" * 10)
    print("SVM Evaluation")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC      : {roc_auc:.4f}")
    print("+" * 10)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc
    }
