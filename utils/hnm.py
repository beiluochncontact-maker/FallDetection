"""Hard Negative Mining helpers for LOSO training."""

from __future__ import annotations

import numpy as np

import config


def task_id_from_trial(trial: str) -> int:
    """T02R05 -> 2"""
    return int(str(trial)[1:3])


def flatten_train_arrays(train_set: dict):
    """Return X, y, trials from LOSO train_set dict."""
    x_train = []
    y_train = []
    trials = []

    for subject_samples in train_set.values():
        for sample in subject_samples:
            x_train.append(sample["feature"])
            y_train.append(sample["label"])
            trials.append(sample["trial"])

    return (
        np.asarray(x_train, dtype=np.float32),
        np.asarray(y_train, dtype=np.int32),
        np.asarray(trials),
    )


def mine_hard_negatives(
    x_train: np.ndarray,
    y_train: np.ndarray,
    trials: np.ndarray,
    pilot,
    hard_fraction: float | None = None,
    adl_bonus: float | None = None,
    hard_tasks: tuple[int, ...] | None = None,
    random_state: int = 42,
):
    """
    Rebuild a class-balanced training set whose negatives are mostly
    hard (high pilot P(fall), with optional ADL bonus).

    Returns
    -------
    x_bal, y_bal, info_dict
    """
    hard_fraction = (
        config.HNM_HARD_FRACTION if hard_fraction is None else hard_fraction
    )
    adl_bonus = config.HNM_ADL_BONUS if adl_bonus is None else adl_bonus
    hard_tasks = (
        tuple(config.HNM_HARD_ADL_TASKS) if hard_tasks is None else hard_tasks
    )

    pos_mask = y_train == 1
    neg_mask = y_train == 0
    x_pos = x_train[pos_mask]
    x_neg = x_train[neg_mask]
    trials_neg = trials[neg_mask]
    n_pos = int(len(x_pos))
    n_neg = int(len(x_neg))

    if n_pos == 0 or n_neg == 0:
        raise ValueError("HNM requires both fall and non-fall train windows")

    proba = pilot.predict_proba(x_neg)[:, 1]
    tasks = np.asarray([task_id_from_trial(t) for t in trials_neg], dtype=int)
    is_hard_adl = np.isin(tasks, hard_tasks)
    mine_score = proba + adl_bonus * is_hard_adl.astype(np.float64)

    order = np.argsort(-mine_score)
    n_hard = int(round(n_pos * hard_fraction))
    n_hard = max(1, min(n_hard, n_neg, n_pos))
    n_easy = n_pos - n_hard

    hard_sel = order[:n_hard]
    remain = order[n_hard:]
    rng = np.random.RandomState(random_state)

    if n_easy > 0:
        if len(remain) >= n_easy:
            easy_sel = rng.choice(remain, size=n_easy, replace=False)
        elif len(remain) > 0:
            easy_sel = rng.choice(remain, size=n_easy, replace=True)
        else:
            easy_sel = rng.choice(hard_sel, size=n_easy, replace=True)
        neg_sel = np.concatenate([hard_sel, easy_sel])
    else:
        neg_sel = hard_sel

    rng.shuffle(neg_sel)
    x_neg_bal = x_neg[neg_sel]
    x_bal = np.concatenate([x_pos, x_neg_bal], axis=0)
    y_bal = np.concatenate(
        [
            np.ones(n_pos, dtype=np.int32),
            np.zeros(len(x_neg_bal), dtype=np.int32),
        ]
    )

    info = {
        "n_pos": n_pos,
        "n_neg_selected": int(len(neg_sel)),
        "n_hard": int(n_hard),
        "n_easy": int(n_easy),
        "hard_adl_in_selected": int(is_hard_adl[neg_sel].sum()),
        "mine_score_p50": float(np.median(mine_score[neg_sel])),
        "mine_score_p90": float(np.quantile(mine_score[neg_sel], 0.9)),
        "pilot_p_p50": float(np.median(proba[neg_sel])),
        "pilot_p_p90": float(np.quantile(proba[neg_sel], 0.9)),
    }
    return x_bal, y_bal, info
