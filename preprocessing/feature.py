import numpy as np
import config
from scipy.stats import skew, kurtosis
from collections import defaultdict


def extract_basic_statistics(signal):

    features = []
    std = np.std(signal)
    if std < 1e-8:
        skewness = 0.0
        kurt = 0.0
    else:
        skewness = skew(signal)
        kurt = kurtosis(signal)


    features.extend([
        np.mean(signal),
        std,
        np.max(signal),
        np.min(signal),
        np.max(signal) - np.min(signal),                    # range
        np.ptp(signal),                                     # Peak-to-Peak
        np.sqrt(np.mean(signal ** 2)),                      # RMS
        np.var(signal),                                     # variance
        np.median(signal),                                  
        np.percentile(signal, 75) - np.percentile(signal, 25), #IQR
        np.sum(signal ** 2),                                # energy
        skewness,                                           # skewness
        kurt                                                # kurtosis
    ])

    return features


def check_invalid_features(features, feature_names, sample=None):

    features = np.asarray(features, dtype=np.float32)

    nan_index = np.where(np.isnan(features))[0]
    inf_index = np.where(np.isinf(features))[0]

    if len(nan_index) > 0 or len(inf_index) > 0:

        print("=" * 60)

        if sample is not None:
            print(f"Subject : {sample['subject']}")
            print(f"Trial   : {sample['trial']}")

        '''  error message
        if len(nan_index) > 0:
            print("\nNaN Features:")

            for idx in nan_index:
                print(f"[{idx:3d}] {feature_names[idx]} = NaN")

        if len(inf_index) > 0:
            print("\nInf Features:")

            for idx in inf_index:
                print(f"[{idx:3d}] {feature_names[idx]} = Inf")
        '''


    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return features


def feature_extractor(windows):

    feature_dataset = []
    for sample in windows:

        data = sample["data"]      # (128,9)
        features = []

        # Acc Gyr Euler
        for i in range(data.shape[1]):
            features.extend(
                extract_basic_statistics(data[:, i])
            )

        # Magnitude
        acc_mag = np.linalg.norm(data[:, 0:3], axis=1)
        gyr_mag = np.linalg.norm(data[:, 3:6], axis=1)
        euler_mag = np.linalg.norm(data[:, 6:9], axis=1)

        features.extend(extract_basic_statistics(acc_mag))
        features.extend(extract_basic_statistics(gyr_mag))
        features.extend(extract_basic_statistics(euler_mag))

        # Signal Magnitude Area(SMA)
        acc_sma = np.mean(np.abs(data[:, 0]) +
                          np.abs(data[:, 1]) +
                          np.abs(data[:, 2]))
        gyr_sma = np.mean(np.abs(data[:, 3]) +
                          np.abs(data[:, 4]) +
                          np.abs(data[:, 5]))

        features.extend([
            acc_sma,
            gyr_sma
        ])

        features = check_invalid_features(
            features,
            config.FEATURE_LIST,
            sample
        )

        # Feature dataset
        feature_dataset.append({
            "subject": sample["subject"],
            "trial": sample["trial"],
            "label": sample["label"],
            "start": sample["start"],
            "end": sample["end"],
            "window": sample["data"],                               # Original Window Data
            "feature": np.array(features, dtype=np.float32)         # Feature Vector
        })

    return feature_dataset



def group_features(feature_dataset):

    #Group all window features by (subject, trial).
    all_features = defaultdict(lambda: {
        "subject": None,
        "trial": None,
        "label": None,
        "windows": []
    })

    for sample in feature_dataset:

        key = (sample["subject"], sample["trial"])

        all_features[key]["subject"] = sample["subject"]
        all_features[key]["trial"] = sample["trial"]
        all_features[key]["label"] = sample["label"]

        all_features[key]["windows"].append({

            "feature": sample["feature"],
            "window": sample.get("window", None),
            "start": sample.get("start", None),
            "end": sample.get("end", None)
        })

    for sample in all_features.values():

        sample["windows"].sort(
            key=lambda x: x["start"] if x["start"] is not None else 0
        )

    return dict(all_features)