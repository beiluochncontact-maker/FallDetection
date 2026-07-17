from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import OUTPUT_DIR
from visualization.base import BaseVisualizer


class ResultVisualizer(BaseVisualizer):

    def __init__(self, model_name):
        save_dir = Path(OUTPUT_DIR) / "figures" / "results" / model_name
        super().__init__(save_dir)
        self.result_dir = Path(OUTPUT_DIR) / model_name
        self.model_name = model_name

    def load_loso(self):
        return pd.read_csv(self.result_dir / "loso_results.csv")

    def load_confusion(self):
        return np.load(self.result_dir / "confusion_matrix.npy")

    def load_roc(self):
        data = np.load(self.result_dir / "roc_curve.npz")
        return data["fpr"], data["tpr"], float(data["auc"])

    def plot_confusion_matrix(self):
        cm = self.load_confusion()

        fig = plt.figure(figsize=(5, 5))
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()

        labels = ["Normal", "Fall"]
        plt.xticks([0, 1], labels)
        plt.yticks([0, 1], labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")

        for i in range(2):
            for j in range(2):
                plt.text(
                    j,
                    i,
                    str(int(cm[i, j])),
                    ha="center",
                    va="center",
                    fontsize=14,
                    color="black",
                )

        plt.title(f"Confusion Matrix ({self.model_name})")
        self.save(fig, "confusion_matrix.png")

    def plot_roc_curve(self):
        roc_path = self.result_dir / "roc_curve.npz"
        if not roc_path.exists():
            return

        fpr, tpr, roc_auc = self.load_roc()

        fig = plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve ({self.model_name})")
        plt.legend()
        self.save(fig, "roc_curve.png")

    def plot_loso_prf(self):
        df = self.load_loso()
        x = df["Fold"] if "Fold" in df.columns else np.arange(1, len(df) + 1)

        fig = plt.figure(figsize=(12, 5))
        for col, color in (
            ("Precision", "#2980b9"),
            ("Recall", "#27ae60"),
            ("F1", "#c0392b"),
        ):
            if col not in df.columns:
                continue
            plt.plot(
                x,
                df[col],
                marker="o",
                markersize=3,
                linewidth=1.2,
                label=col,
                color=color,
            )

        plt.xlabel("LOSO Fold")
        plt.ylabel("Score")
        plt.title(f"LOSO Precision / Recall / F1 ({self.model_name})")
        plt.legend()
        plt.grid(True, alpha=0.3)

        vals = []
        for col in ("Precision", "Recall", "F1"):
            if col in df.columns:
                vals.extend(df[col].dropna().tolist())
        if vals:
            lo, hi = min(vals), max(vals)
            pad = max(0.02, (hi - lo) * 0.15)
            plt.ylim(max(0.0, lo - pad), min(1.05, hi + pad))

        if "Subject" in df.columns and len(df) <= 40:
            plt.xticks(x, df["Subject"], rotation=90, fontsize=7)

        self.save(fig, "loso_prf.png")

    def visualize_all(self):
        print(f"Generating Result Figures ({self.model_name})...")
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        self.plot_loso_prf()
        print("Finished.")
