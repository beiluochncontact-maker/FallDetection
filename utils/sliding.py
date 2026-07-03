import config
import numpy as np

def sliding_window(features,
                   window_size=config.WINDOW_SIZE,
                   stride=config.STRIDE):

    windows = []

    for start in range(
            0,
            len(features)-window_size+1,
            stride):

        windows.append(
            features[start:start+window_size]
        )

    return windows


def vote_prediction(
        window_predictions,
        vote_size=config.VOTE_SIZE,
        threshold=config.VOTE_THRESHOLD):

    for i in range(
            len(window_predictions)-vote_size+1):

        votes = window_predictions[
            i:i+vote_size
        ]

        ratio = np.mean(votes)

        if ratio >= threshold:
            return 1

    return 0