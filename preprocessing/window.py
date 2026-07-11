import numpy as np
import pandas as pd
import config

rng = np.random.default_rng(42)


def build_windows(dataset):

    all_windows = []
    for sample in dataset.values():

        df = pd.read_csv(sample["csv_path"])
        total_length = len(df)

        window_size = config.WINDOW_SIZE

        # Window building
        if sample["is_fall"]:

            onset = sample["onset"]
            impact = sample["impact"]
            event_span = int(window_size * (1 - config.ACCEPT_RATE))

            start = onset - event_span
            start = max(0, start)
            end = impact + event_span
            end = min(total_length, end)

            
            if start + config.WINDOW_SIZE > total_length:
                start = total_length - config.WINDOW_SIZE
            
            label = 1

        
        else:
            onset = None
            impact = None
            start = 0
            end = total_length
            label = 0

        data = (
            df.iloc[start:end][config.FEATURE_COLUMNS]
            .to_numpy(dtype=np.float32)
        )
                

        # Generate Sliding Windows
        stride = config.STRIDE
        starts = list(range(0, len(data)-window_size+1, stride))

        last_start = max(0, len(data)-window_size)

        if starts[-1] != last_start:
            starts.append(last_start)

        for s in starts:

            window = data[s:s + window_size]

            all_windows.append({
                "sample_id": sample["subject"] + sample["trial"],
                "subject": sample["subject"],
                "trial": sample["trial"],
                "label": label,
                "start": s,
                "end": s + window_size - 1,
                "data": window
            })


    return all_windows


# Subwindows
def split_subwindows(window, window_size):

    sub_size = int(window_size * 0.25)
    step = int(window_size * 0.125)

    subwindows = []

    start = 0
    while len(subwindows) < 7:

        end = start + sub_size

        if end > len(window):
            break

        sub = window[start:end]
        subwindows.append(sub)

        start += step

    return subwindows


# for Transformer
def build_hierarchical_window(window):

    data = window["data"]
    subwindows = split_subwindows(
        data,
        config.WINDOW_SIZE
    )

    return {
        **window,
        "subwindows": subwindows
    }