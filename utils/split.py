from sklearn.model_selection import train_test_split


def split_loso(dataset, random_state=42):
    # Subjects
    subjects = sorted({
        sample["subject"]
        for sample in dataset
    })

    # LOSO
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
        train_set = [
            sample
            for sample in dataset
            if sample["subject"] in train_subjects
        ]


        # Validation Set
        val_set = [
            sample
            for sample in dataset
            if sample["subject"] in val_subjects
        ]


        test_set = [
            sample
            for sample in dataset
            if sample["subject"] == test_subject
        ]

        yield {
            "test_subject": test_subject,
            "train_set": train_set,
            "val_set": val_set,
            "test_set": test_set
        }