import numpy as np

from joblib import dump

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
        param_grid):

    x_train = np.array([
        sample["feature"]
        for sample in train_set
    ])

    # test NaN
    for i, sample in enumerate(train_set):

        feature = sample["feature"]
        if np.isnan(feature).any():
            print("=" * 50)
            print("NaN Sample")
            print("Index:", i)
            print("Subject:", sample["subject"])
            print("Trial:", sample["trial"])


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
    best_params = None

    for params in ParameterGrid(param_grid):

        model = Pipeline([
        
            (
                "scaler",
                StandardScaler()
            ),
        
            (
                "svm",
                SVC(
                    **params,
                    class_weight="balanced",
                    random_state=42
                )
            )
        
        ])

        print("Train labels:", np.unique(y_train, return_counts=True))
        print("Train subjects:")
        subjects = sorted(set(sample["subject"] for sample in train_set))
        print(subjects)
        model.fit(
            x_train,
            y_train
        )
        print("Fall samples:")
        print(sum(sample["label"] for sample in train_set))


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
            best_params = params

    return best_model


def evaluate_svm(
        model,
        test_set):

    x_test = np.array([
        sample["feature"]
        for sample in test_set
    ])

    y_test = np.array([
        sample["label"]
        for sample in test_set
    ])

    pred = model.predict(
        x_test
    )

    result = {

        "accuracy": accuracy_score(
            y_test,
            pred
        ),

        "precision": precision_score(
            y_test,
            pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_test,
            pred,
            zero_division=0
        ),

        "f1": f1_score(
            y_test,
            pred,
            zero_division=0
        ),

        "confusion_matrix": confusion_matrix(
            y_test,
            pred
        )

    }

    return result