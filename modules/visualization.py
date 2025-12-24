import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)


def generate_all_plots(
    df_results: pd.DataFrame,
    output_dir: str,
    final_thr: Optional[float] = None,
    p_thr: Optional[float] = None,
    iqr_thr: Optional[float] = None,
    window_minutes: Optional[int] = None,
) -> str:
    """
    Генерирует набор графиков для ВКР по результатам детектора.
    Вся визуализация включается одним вызовом из main.py.

    Ожидаемые колонки в df_results:
      - time (желательно datetime)
      - anomaly_score
      - final_anomaly (0/1)
      - destination_port (для топ-портов)
      - packet_count, bytes_sent (для scatter)
      - label (0/1) опционально — для confusion matrix и PR-кривой

    Возвращает путь к папке с графиками.
    """
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    _plot_score_distribution(df_results, plots_dir, final_thr, p_thr, iqr_thr)
    _plot_anomalies_timeline(df_results, plots_dir, window_minutes=window_minutes)
    _plot_top_destination_ports(df_results, plots_dir, top_n=15)
    _plot_confusion_matrix_if_available(df_results, plots_dir)
    _plot_pr_curve_if_available(df_results, plots_dir)
    _plot_scatter_bytes_packets(df_results, plots_dir, max_points=6000)

    return plots_dir


# -----------------------------
# Plot 1: Score distribution + thresholds
# -----------------------------
def _plot_score_distribution(
    df: pd.DataFrame,
    out_dir: str,
    final_thr: Optional[float],
    p_thr: Optional[float],
    iqr_thr: Optional[float],
):
    if "anomaly_score" not in df.columns:
        print("[PLOTS] skip: нет колонки anomaly_score")
        return

    scores = pd.to_numeric(df["anomaly_score"], errors="coerce").dropna().values
    if len(scores) == 0:
        print("[PLOTS] skip: anomaly_score пустой/NaN")
        return

    plt.figure(figsize=(10, 5))
    plt.hist(scores, bins=50)

    # Пороги рисуем только если они переданы
    if p_thr is not None:
        plt.axvline(p_thr, linestyle="--", linewidth=2, label=f"p-threshold: {p_thr:.4f}")
    if iqr_thr is not None:
        plt.axvline(iqr_thr, linestyle="--", linewidth=2, label=f"IQR-threshold: {iqr_thr:.4f}")
    if final_thr is not None:
        plt.axvline(final_thr, linestyle="-", linewidth=2, label=f"Final threshold: {final_thr:.4f}")

    plt.title("Распределение anomaly_score (Isolation Forest) и пороги")
    plt.xlabel("anomaly_score (меньше = более аномально)")
    plt.ylabel("Количество")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(out_dir, "01_scores_distribution.png")
    plt.savefig(path, dpi=200)
    plt.close()


# -----------------------------
# Plot 2: Timeline anomalies count
# -----------------------------
def _plot_anomalies_timeline(df: pd.DataFrame, out_dir: str, window_minutes: Optional[int] = None):
    if "time" not in df.columns:
        print("[PLOTS] skip: нет колонки time")
        return
    if "final_anomaly" not in df.columns:
        print("[PLOTS] skip: нет колонки final_anomaly")
        return

    temp = df[["time", "final_anomaly"]].copy()
    temp["time"] = pd.to_datetime(temp["time"], errors="coerce")
    temp = temp.dropna(subset=["time"])

    if len(temp) == 0:
        print("[PLOTS] skip: time пустой/NaN")
        return

    # Если уже есть time_window (из frequency_analysis) — используем его.
    if "time_window" in df.columns:
        temp["bucket"] = pd.to_datetime(df["time_window"], errors="coerce")
    else:
        # Иначе сами бьём по окнам.
        freq = f"{int(window_minutes)}min" if window_minutes else "1min"
        temp["bucket"] = temp["time"].dt.floor(freq)

    grp = temp.groupby("bucket")["final_anomaly"].sum().sort_index()

    plt.figure(figsize=(12, 5))
    plt.plot(grp.index, grp.values)
    plt.title("Динамика: число аномалий по времени (final_anomaly)")
    plt.xlabel("Время")
    plt.ylabel("Количество аномалий в окне")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    path = os.path.join(out_dir, "02_anomalies_timeline.png")
    plt.savefig(path, dpi=200)
    plt.close()


# -----------------------------
# Plot 3: Top destination ports (anomaly vs normal)
# -----------------------------
def _plot_top_destination_ports(df: pd.DataFrame, out_dir: str, top_n: int = 15):
    if "destination_port" not in df.columns:
        print("[PLOTS] skip: нет колонки destination_port")
        return
    if "final_anomaly" not in df.columns:
        print("[PLOTS] skip: нет колонки final_anomaly")
        return

    tmp = df[["destination_port", "final_anomaly"]].copy()
    tmp["destination_port"] = pd.to_numeric(tmp["destination_port"], errors="coerce")
    tmp = tmp.dropna(subset=["destination_port"])

    if len(tmp) == 0:
        print("[PLOTS] skip: destination_port пустой/NaN")
        return

    anom = tmp[tmp["final_anomaly"].astype(int) == 1]
    norm = tmp[tmp["final_anomaly"].astype(int) == 0]

    anom_counts = anom["destination_port"].value_counts().head(top_n)
    ports = anom_counts.index.astype(int).tolist()

    norm_counts = norm["destination_port"].value_counts().reindex(ports).fillna(0).astype(int)

    x = np.arange(len(ports))
    w = 0.4

    plt.figure(figsize=(12, 5))
    plt.bar(x - w / 2, anom_counts.values, width=w, label="Аномалии")
    plt.bar(x + w / 2, norm_counts.values, width=w, label="Норма")

    plt.title(f"Top-{top_n} destination_port: аномалии vs норма")
    plt.xlabel("destination_port")
    plt.ylabel("Количество")
    plt.xticks(x, [str(p) for p in ports], rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(out_dir, "03_top_destination_ports.png")
    plt.savefig(path, dpi=200)
    plt.close()


# -----------------------------
# Plot 4: Confusion matrix (if label exists)
# -----------------------------
def _plot_confusion_matrix_if_available(df: pd.DataFrame, out_dir: str):
    if "label" not in df.columns:
        print("[PLOTS] skip confusion: нет колонки label")
        return
    if "final_anomaly" not in df.columns:
        print("[PLOTS] skip confusion: нет колонки final_anomaly")
        return

    y_true = pd.to_numeric(df["label"], errors="coerce").dropna().astype(int).values
    y_pred = pd.to_numeric(df.loc[df["label"].notna(), "final_anomaly"], errors="coerce").fillna(0).astype(int).values

    if len(y_true) == 0 or len(y_true) != len(y_pred):
        print("[PLOTS] skip confusion: некорректные y_true/y_pred")
        return

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Матрица ошибок (Confusion Matrix)")
    plt.xticks([0, 1], ["Pred: 0", "Pred: 1"])
    plt.yticks([0, 1], ["True: 0", "True: 1"])

    # Подписи ячеек
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center", fontsize=12)

    plt.tight_layout()

    path = os.path.join(out_dir, "04_confusion_matrix.png")
    plt.savefig(path, dpi=200)
    plt.close()

    # Дополнительно: краткий текстовый лог (удобно для ВКР)
    txt_path = os.path.join(out_dir, "04_confusion_matrix_values.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}\n")


# -----------------------------
# Plot 5: PR curve (if label exists)
# -----------------------------
def _plot_pr_curve_if_available(df: pd.DataFrame, out_dir: str):
    if "label" not in df.columns:
        print("[PLOTS] skip PR: нет колонки label")
        return
    if "anomaly_score" not in df.columns:
        print("[PLOTS] skip PR: нет anomaly_score")
        return

    mask = df["label"].notna()
    y_true = pd.to_numeric(df.loc[mask, "label"], errors="coerce").dropna().astype(int).values

    # score_samples: меньше = аномальнее -> для PR сделаем "чем больше, тем аномальнее"
    scores = pd.to_numeric(df.loc[mask, "anomaly_score"], errors="coerce").dropna().values
    if len(y_true) == 0 or len(scores) == 0 or len(y_true) != len(scores):
        print("[PLOTS] skip PR: некорректные y_true/scores")
        return

    y_score = -scores
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision)
    plt.title(f"Precision–Recall кривая по anomaly_score (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()

    path = os.path.join(out_dir, "05_precision_recall_curve.png")
    plt.savefig(path, dpi=200)
    plt.close()


# -----------------------------
# Plot 6: Scatter (packets vs bytes) colored by anomaly flag
# -----------------------------
def _plot_scatter_bytes_packets(df: pd.DataFrame, out_dir: str, max_points: int = 6000):
    needed = {"packet_count", "bytes_sent", "final_anomaly"}
    if not needed.issubset(df.columns):
        print("[PLOTS] skip scatter: нет нужных колонок (packet_count, bytes_sent, final_anomaly)")
        return

    tmp = df[list(needed)].copy()
    tmp["packet_count"] = pd.to_numeric(tmp["packet_count"], errors="coerce")
    tmp["bytes_sent"] = pd.to_numeric(tmp["bytes_sent"], errors="coerce")
    tmp["final_anomaly"] = pd.to_numeric(tmp["final_anomaly"], errors="coerce").fillna(0).astype(int)
    tmp = tmp.dropna(subset=["packet_count", "bytes_sent"])

    if len(tmp) == 0:
        print("[PLOTS] skip scatter: данные пустые")
        return

    # Сэмплинг для скорости/читабельности
    if len(tmp) > max_points:
        tmp = tmp.sample(n=max_points, random_state=42)

    normal = tmp[tmp["final_anomaly"] == 0]
    anom = tmp[tmp["final_anomaly"] == 1]

    plt.figure(figsize=(8, 6))
    plt.scatter(normal["packet_count"], normal["bytes_sent"], alpha=0.25, s=10, label="Норма")
    plt.scatter(anom["packet_count"], anom["bytes_sent"], alpha=0.9, s=18, marker="x", label="Аномалии")

    plt.title("Scatter: packet_count vs bytes_sent (по final_anomaly)")
    plt.xlabel("packet_count")
    plt.ylabel("bytes_sent")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(out_dir, "06_scatter_packets_bytes.png")
    plt.savefig(path, dpi=200)
    plt.close()
