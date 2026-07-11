from sklearn.model_selection import train_test_split
from collections import defaultdict


def build_subject_features(dataset):

    subject_features = defaultdict(list)

    for sample in dataset:
        entry = {
            "subject": sample["subject"],
            "trial": sample["trial"],
            "label": sample["label"],
            "feature": sample["feature"],
            "start": sample["start"],
            "end": sample["end"],
        }

        # Keep raw window for models that need it (e.g. Transformer)
        if "window" in sample:
            entry["window"] = sample["window"]

        subject_features[sample["subject"]].append(entry)

    return dict(subject_features)


def split_loso(train_feature_dataset, test_feature_dataset, random_state=42):
    """
    train/val: features from sampled (train-mode) windows
    test_set : features from dense sliding windows on original sequences
    """
    subject_train = build_subject_features(train_feature_dataset)
    subject_test = build_subject_features(test_feature_dataset)
    subjects = sorted(subject_train.keys())

    loso_dataset = []
    for test_subject in subjects:
        remain_subjects = [
            s for s in subjects if s != test_subject
        ]

        train_subjects, val_subjects = train_test_split(
            remain_subjects,
            test_size=3,
            random_state=random_state,
            shuffle=True
        )

        train_subjects = set(train_subjects)
        val_subjects = set(val_subjects)

        train_set = {
            s: subject_train[s]
            for s in train_subjects
        }

        val_set = {
            s: subject_train[s]
            for s in val_subjects
        }

        # Dense sliding-window features on the original trial sequences
        test_set = subject_test[test_subject]

        loso_dataset.append({
            "test_subject": test_subject,
            "train_set": train_set,
            "val_set": val_set,
            "test_set": test_set
        })

    return loso_dataset
