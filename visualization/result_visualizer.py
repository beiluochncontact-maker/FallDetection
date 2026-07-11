from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import OUTPUT_DIR
from visualization.base import BaseVisualizer


class ResultVisualizer(BaseVisualizer):

    def __init__(self, model_name):
        
        save_dir = Path(OUTPUT_DIR) / "figures" / "results" / model_name

        super().__init__(save_dir)

        self.result_dir = Path("output") / model_name
    # -------------------------------------------------
    # 读取 summary.csv
    # -------------------------------------------------
    def load_summary(self):

        return pd.read_csv(
            self.result_dir / "summary.csv"
        )

    # -------------------------------------------------
    # 读取 loso_results.csv
    # -------------------------------------------------
    def load_loso(self):

        return pd.read_csv(
            self.result_dir / "loso_results.csv"
        )

    # -------------------------------------------------
    # 读取 confusion_matrix.npy
    # -------------------------------------------------
    def load_confusion(self):

        path = self.result_dir / "confusion_matrix.npy"

        try:

            return np.load(path)

        except:

            with open(path, "r") as f:
                return np.array(json.load(f))

    # -------------------------------------------------
    # 读取 feature_importance.npy
    # -------------------------------------------------
    def load_feature_importance(self):

        return np.load(
            self.result_dir / "feature_importance.npy"
        )
    
    # -------------------------------------------------
    # 混淆矩阵
    # -------------------------------------------------
    def plot_confusion_matrix(self):

        cm = self.load_confusion()

        fig = plt.figure(figsize=(5,5))

        plt.imshow(cm, cmap="Blues")

        plt.colorbar()

        labels = ["Normal","Fall"]

        plt.xticks([0,1],labels)
        plt.yticks([0,1],labels)

        plt.xlabel("Predicted")
        plt.ylabel("True")

        for i in range(2):
            for j in range(2):

                plt.text(
                    j,
                    i,
                    str(cm[i,j]),
                    ha="center",
                    va="center",
                    fontsize=14,
                    color="black"
                )

        plt.title("Confusion Matrix")

        self.save(fig,"confusion_matrix.png")

    # -------------------------------------------------
    # 四项指标
    # -------------------------------------------------
    def plot_metrics(self):

        df=self.load_summary()

        metrics=[

            "Accuracy Mean",

            "Precision Mean",

            "Recall Mean",

            "F1 Mean"

        ]

        values=df.loc[0,metrics]

        fig=plt.figure(figsize=(7,4))

        plt.bar(

            ["Accuracy","Precision","Recall","F1"],

            values

        )

        plt.ylim(0.95,1.0)

        plt.ylabel("Score")

        plt.title("Overall Performance")

        self.save(fig,"metrics.png")

    # -------------------------------------------------
    # LOSO
    # -------------------------------------------------
    def plot_loso_accuracy(self):

        df=self.load_loso()

        fig=plt.figure(figsize=(12,6))

        plt.bar(

            df["Subject"],

            df["Accuracy"]

        )

        plt.xticks(rotation=90)

        plt.ylim(0.95,1.0)

        plt.ylabel("Accuracy")

        plt.title("LOSO Accuracy")

        self.save(fig,"loso_accuracy.png")

    # -------------------------------------------------
    # 指标表
    # -------------------------------------------------
    def plot_metric_table(self):

        df=self.load_summary()

        fig,ax=plt.subplots(figsize=(8,2))

        ax.axis("off")

        table=ax.table(

            cellText=np.round(df.values,4),

            colLabels=df.columns,

            loc="center"

        )

        table.scale(1.2,2)

        self.save(fig,"metrics_table.png")
    
    # -------------------------------------------------
    # 特征重要性
    # -------------------------------------------------
    def plot_feature_importance(self):

        importance = self.load_feature_importance()

        # 取前20个最重要的特征
        top_k = 20

        index = np.argsort(importance)[::-1][:top_k]

        values = importance[index]

        fig = plt.figure(figsize=(10,6))

        plt.bar(
            range(top_k),
            values
        )

        plt.xticks(
            range(top_k),
            index,
            rotation=45
        )

        plt.xlabel("Feature Index")

        plt.ylabel("Importance")

        plt.title(f"Top 20 Feature Importance ({self.result_dir.name})")

        self.save(
            fig,
            "feature_importance.png"
        )

    def load_roc(self):

        data = np.load(
            self.result_dir / "roc_curve.npz"
        )

        return (
            data["fpr"],
            data["tpr"],
            data["auc"]
        )
    
    def plot_roc_curve(self):

        fpr, tpr, roc_auc = self.load_roc()

        fig = plt.figure(figsize=(6,6))

        plt.plot(
            fpr,
            tpr,
            label=f"AUC = {roc_auc:.3f}"
        )

        plt.plot(
            [0,1],
            [0,1],
            "--"
        )

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()

        self.save(
            fig,
            "roc_curve.png"
        )

    def plot_train_params(self):

        path = self.result_dir / "train_params.csv"

        if not path.exists():
            return

        df = pd.read_csv(path)

        display = df.copy()
        if "BestParam" in display.columns:
            display["BestParam"] = display["BestParam"].astype(str).str.slice(0, 48)

        fig, ax = plt.subplots(figsize=(12, max(2, 0.35 * len(display) + 1)))
        ax.axis("off")

        table = ax.table(
            cellText=display.values,
            colLabels=display.columns,
            loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.4)

        self.save(fig, "train_params.png")

    # -------------------------------------------------
    def visualize_all(self):

        print("Generating Result Figures...")

        self.plot_confusion_matrix()

        self.plot_metrics()

        self.plot_loso_accuracy()

        self.plot_metric_table()

        self.plot_train_params()

        if (self.result_dir / "roc_curve.npz").exists():
            self.plot_roc_curve()

        if (self.result_dir / "feature_importance.npy").exists():
            self.plot_feature_importance()

        print("Finished.")