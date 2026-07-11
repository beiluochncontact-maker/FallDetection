import numpy as np
import pandas as pd
import config


def _candidate_starts(data_len, window_size, stride):
    if data_len < window_size:
        return []

    starts = list(range(0, data_len - window_size + 1, stride))
    last_start = data_len - window_size

    if not starts:
        return [last_start]

    if starts[-1] != last_start:
        starts.append(last_start)

    return starts


def _sample_starts(starts, max_windows):
    if max_windows is None or len(starts) <= max_windows:
        return starts

    # Uniform coverage along the trial (deterministic)
    indices = np.linspace(0, len(starts) - 1, max_windows)
    indices = np.unique(np.round(indices).astype(int))
    return [starts[i] for i in indices]


def build_windows(dataset, mode="train"):
    """
    mode="train":
        fall     -> event-centered crop + sliding windows
        non-fall -> full trial, uniformly subsampled windows
    mode="test":
        all trials -> full original sequence + dense sliding windows
    """
    if mode not in {"train", "test"}:
        raise ValueError(f"Unsupported window mode: {mode}")

    all_windows = []
    window_size = config.WINDOW_SIZE
    stride = (
        config.TEST_STRIDE if mode == "test" else config.TRAIN_STRIDE
    )

    for sample in dataset.values():
        df = pd.read_csv(sample["csv_path"])
        total_length = len(df)
        label = 1 if sample["is_fall"] else 0

        if mode == "test":
            seg_start, seg_end = 0, total_length
        elif sample["is_fall"]:
            event_span = int(window_size * (1 - config.ACCEPT_RATE))
            seg_start = max(0, sample["onset"] - event_span)
            seg_end = min(total_length, sample["impact"] + event_span)
            if seg_start + window_size > total_length:
                seg_start = max(0, total_length - window_size)
        else:
            seg_start, seg_end = 0, total_length

        data = (
            df.iloc[seg_start:seg_end][config.FEATURE_COLUMNS]
            .to_numpy(dtype=np.float32)
        )

        starts = _candidate_starts(len(data), window_size, stride)

        if mode == "train" and not sample["is_fall"]:
            starts = _sample_starts(starts, config.NON_FALL_MAX_WINDOWS)

        for relative_start in starts:
            absolute_start = seg_start + relative_start
            window = data[relative_start:relative_start + window_size]

            all_windows.append({
                "sample_id": sample["subject"] + sample["trial"],
                "subject": sample["subject"],
                "trial": sample["trial"],
                "label": label,
                "start": absolute_start,
                "end": absolute_start + window_size - 1,
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
