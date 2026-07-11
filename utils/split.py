from sklearn.model_selection import train_test_split
from collections import defaultdict

def build_subject_windows(window_dataset):

    subject_windows = defaultdict(list)

    for sample in window_dataset:
        subject_windows[sample["subject"]].append(sample)

    return dict(subject_windows)


def build_subject_features(dataset):
    
    subject_features = defaultdict(list)

    for sample in dataset:
        subject_features[sample["subject"]].append({

            "trial": sample["trial"],
            "label": sample["label"],
            "feature": sample["feature"],
            "start": sample["start"],
            "end": sample["end"]

        })

    # Compose subjects into dictionary
    subject_features = {
        subj: features
        for subj, features in subject_features.items()
    }

    return subject_features


def split_loso(feature_dataset, window_dataset, random_state=42):
    # Subjects
    subject_features = build_subject_features(feature_dataset)
    subject_windows = build_subject_windows(window_dataset)
    subjects = sorted(subject_features.keys())

    # LOSO
    loso_dataset = []
    for test_subject in subjects:
        # Remaining 32 Subjects
        remain_subjects = [
            s
            for s in subjects
            if s != test_subject
        ]


        # Train / Validation Subject
        train_subjects, val_subjects = train_test_split(
            remain_subjects,
            test_size=3,
            random_state=random_state,
            shuffle=True
        )

        train_subjects = set(train_subjects)
        val_subjects = set(val_subjects)

        # Train Set
        train_set = {
            s: subject_features[s]
            for s in train_subjects
        }

        # Validation Set
        val_set = {
            s: subject_features[s]
            for s in val_subjects
        }

        '''
        test_set = {
            "feature": subject_features[test_subject],
            "window": subject_windows[test_subject]
        }
        '''
        test_set = subject_features[test_subject]
        
        loso_dataset.append({
            "test_subject": test_subject,
            "train_set": train_set,
            "val_set": val_set,
            "test_set": test_set
        })

    return loso_dataset