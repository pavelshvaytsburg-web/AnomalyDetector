import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
def plot_anomaly_scores(df: pd.DataFrame, output_path: str):
    """
    Гистограмма распределения anomaly_score.
    Чем левее хвост и чем он “откусаннее”, тем сильнее выражены аномалии.
    """
    plt.figure(figsize=(8, 4))
    plt.hist(df["anomaly_score"], bins=40)
    plt.title("Распределение anomaly_score")
    plt.xlabel("anomaly_score")
    plt.ylabel("Количество записей")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_packets_vs_bytes(df: pd.DataFrame, output_path: str):
    """
    Диаграмма рассеяния packet_count vs bytes_sent
    с выделением аномалий (final_anomaly = 1).
    """
    plt.figure(figsize=(6, 6))

    normal = df[df["final_anomaly"] == 0]
    anomalies = df[df["final_anomaly"] == 1]

    # Норма
    plt.scatter(
        normal["packet_count"],
        normal["bytes_sent"],
        s=10,
        alpha=0.5,
        label="Норма"
    )

    # Аномалии
    plt.scatter(
        anomalies["packet_count"],
        anomalies["bytes_sent"],
        s=30,
        marker="x",
        label="Аномалии"
    )

    plt.title("packet_count vs bytes_sent")
    plt.xlabel("packet_count")
    plt.ylabel("bytes_sent")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
