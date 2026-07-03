import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import OUTPUT_DIR
from visualization.base import BaseVisualizer


class DataVisualizer(BaseVisualizer):

    def __init__(self):

        super().__init__(Path(OUTPUT_DIR) / "figures" / "data")

    # --------------------------------------------------
    # 类别统计
    # --------------------------------------------------
    def plot_label_distribution(self, dataset):

        fall = 0
        normal = 0

        for item in dataset.values():

            if item["is_fall"]:
                fall += 1
            else:
                normal += 1

        fig = plt.figure(figsize=(6, 4))

        plt.bar(
            ["Fall", "Normal"],
            [fall, normal]
        )

        plt.title("Label Distribution")
        plt.ylabel("Number of Samples")

        self.save(fig, "label_distribution.png")

    # --------------------------------------------------
    # 缺失值统计
    # --------------------------------------------------
    def plot_missing_values(self, dataset):

        csv_path = next(iter(dataset.values()))["csv_path"]

        df = pd.read_csv(csv_path)

        missing = df.isnull().sum()

        fig = plt.figure(figsize=(10, 5))

        missing.plot(kind="bar")

        plt.title("Missing Values")

        self.save(fig, "missing_values.png")

    # --------------------------------------------------
    # 随机传感器曲线
    # --------------------------------------------------
    def plot_sensor_signal(self, dataset):

        sample = random.choice(list(dataset.values()))

        df = pd.read_csv(sample["csv_path"])

        columns = [
            "AccX",
            "AccY",
            "AccZ"
        ]

        fig = plt.figure(figsize=(12, 5))

        for col in columns:

            if col in df.columns:
                plt.plot(df[col], label=col)

        plt.title("Accelerometer Signal")

        plt.legend()

        self.save(fig, "sensor_signal.png")

    # --------------------------------------------------
    # 跌倒 VS 非跌倒
    # --------------------------------------------------
    def plot_fall_vs_normal(self, dataset):

        fall = None
        normal = None

        for item in dataset.values():

            if item["is_fall"] and fall is None:
                fall = item

            if (not item["is_fall"]) and normal is None:
                normal = item

            if fall and normal:
                break

        if fall is None or normal is None:
            return

        fall_df = pd.read_csv(fall["csv_path"])
        normal_df = pd.read_csv(normal["csv_path"])

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(12, 8)
        )

        axes[0].plot(fall_df["AccX"])
        axes[0].set_title("Fall Sample")

        axes[1].plot(normal_df["AccX"])
        axes[1].set_title("Normal Sample")

        self.save(fig, "fall_vs_normal.png")
    
    # -------------------------------------------------
    # 数据集划分比例
    # -------------------------------------------------
    def plot_dataset_split(self):

        labels = ["Train", "Validation", "Test"]
        sizes = [80, 10, 10]

        fig = plt.figure(figsize=(6, 6))

        plt.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90
        )

        plt.title("Dataset Split")

        self.save(fig, "dataset_split.png")

    # --------------------------------------------------
    # 自动生成全部图片
    # --------------------------------------------------
    def visualize_all(self, dataset):

        print("\nGenerating Data Visualization...")

        self.plot_label_distribution(dataset)

        self.plot_missing_values(dataset)

        self.plot_sensor_signal(dataset)

        self.plot_fall_vs_normal(dataset)

        self.plot_dataset_split()

        print("Finished.\n")