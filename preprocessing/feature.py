import numpy as np
from scipy.stats import skew, kurtosis


def extract_statistics(signal):

    return [
        np.mean(signal),
        np.std(signal),
        np.max(signal),
        np.min(signal),
        np.max(signal) - np.min(signal),                    # range
        np.ptp(signal),                                     # Peak-to-Peak
        np.sqrt(np.mean(signal ** 2)),                      # RMS
        np.var(signal),                                     # variance
        np.median(signal),                                  
        np.percentile(signal, 75) - np.percentile(signal, 25), #IQR
        np.sum(signal ** 2),                                # energy
        skew(signal),                                       # skewness
        kurtosis(signal)                                    # kurtosis
    ]


def feature_extractor(windows):

    feature_dataset = []
    for sample in windows:

        data = sample["data"]      # (128,9)
        features = []

        # Acc Gyr Euler
        for i in range(data.shape[1]):
            features.extend(
                extract_statistics(data[:, i])
            )

        # Magnitude
        acc_mag = np.linalg.norm(data[:, 0:3], axis=1)
        gyr_mag = np.linalg.norm(data[:, 3:6], axis=1)
        euler_mag = np.linalg.norm(data[:, 6:9], axis=1)

        features.extend(extract_statistics(acc_mag))
        features.extend(extract_statistics(gyr_mag))
        features.extend(extract_statistics(euler_mag))

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

        # Feature dataset
        feature_dataset.append({
            "subject": sample["subject"],
            "trial": sample["trial"],
            "label": sample["label"],
            "feature": np.array(features, dtype=np.float32)
        })

    return feature_dataset