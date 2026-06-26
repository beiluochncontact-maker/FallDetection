import numpy as np
import pandas as pd
import config

rng = np.random.default_rng(42)


def build_windows(dataset):

    windows = []
    for sample in dataset.values():

        df = pd.read_csv(sample["csv_path"])
        total_length = len(df)

        # Window building
        if sample["is_fall"]:

            onset = sample["onset"]
            impact = sample["impact"]
            center = (onset + impact) // 2

            start = center - config.WINDOW_SIZE // 2
            start = max(0, start)

            if start + config.WINDOW_SIZE > total_length:
                start = total_length - config.WINDOW_SIZE

            label = 1

        else:
            # random window for unfallen samples
            start = rng.integers(
                0,
                total_length - config.WINDOW_SIZE + 1
            )

            label = 0

        end = start + config.WINDOW_SIZE

        window = (
            df.iloc[start:end][config.FEATURE_COLUMNS]
            .to_numpy(dtype=np.float32)
        )

        windows.append({
            "sample_id": sample["subject"] + sample["trial"],
            "subject": sample["subject"],
            "trial": sample["trial"],
            "label": label,
            "start": start,
            "end": end - 1,
            "data": window
        })

    return windows