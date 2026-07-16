import config
import numpy as np

def sliding_window(features,
                   window_size=config.WINDOW_SIZE,
                   stride=config.TRAIN_STRIDE):

    windows = []

    for start in range(
            0,
            len(features)-window_size+1,
            stride):

        windows.append(
            features[start:start+window_size]
        )

    return windows


def trial_score_from_probs(
        window_probabilities,
        smooth_size=None,
        score_agg=None,
        score_alpha=None):
    """
    Aggregate window-level fall probabilities into a trial score.

    score_agg:
      - "mean" (default): max over sliding means of length k
      - "max_alpha_min": for each adjacent pair (k=2), use max+α·min,
        then take the max over pairs. α may be negative.
      For k==1 (or a single window), always returns max(p).
    """
    probs = np.asarray(window_probabilities, dtype=np.float64)

    if probs.size == 0:
        return 0.0

    k = config.PROB_SMOOTH_SIZE if smooth_size is None else smooth_size
    k = max(1, int(k))
    agg = config.SCORE_AGG if score_agg is None else score_agg
    alpha = config.SCORE_ALPHA if score_alpha is None else score_alpha

    if k == 1 or probs.size < k:
        return float(np.max(probs))

    if agg == "max_alpha_min":
        # Sliding windows of length k: max + α·min, then take the max over starts.
        # k=2 reduces to the previous adjacent-pair definition.
        scores = []
        for i in range(probs.size - k + 1):
            seg = probs[i : i + k]
            scores.append(float(np.max(seg) + float(alpha) * np.min(seg)))
        return float(np.max(scores))

    kernel = np.ones(k, dtype=np.float64) / k
    sliding_means = np.convolve(probs, kernel, mode="valid")
    return float(np.max(sliding_means))


def trial_decision_from_probs(
        window_probabilities,
        smooth_size=None,
        threshold=None,
        score_agg=None,
        score_alpha=None):
    """
    Return (trial_prediction, trial_score) from window probabilities.
    Prediction and score share the same aggregation so metrics stay aligned.
    """
    score = trial_score_from_probs(
        window_probabilities,
        smooth_size=smooth_size,
        score_agg=score_agg,
        score_alpha=score_alpha,
    )
    tau = config.PROB_THRESHOLD if threshold is None else threshold
    prediction = int(score >= tau)
    return prediction, score
