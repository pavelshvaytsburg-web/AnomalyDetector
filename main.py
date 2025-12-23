import argparse
import os

import pandas as pd

from modules.data_loader import load_data
from modules.model import (
    train_isolation_forest,
    anomaly_scores,
    detect_anomalies_custom,
    build_results_dataframe,
)
from modules.frequency_analysis import add_frequency_features
from modules.evaluation import compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Unsupervised ML Anomaly Detector (simple, tuning-focused)")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="output")

    # Частотный анализ
    parser.add_argument("--window", type=int, default=1, help="окно частотного анализа в минутах (1/5)")
    parser.add_argument(
        "--freq_key",
        choices=["dst_port", "dst_port_proto"],
        default="dst_port",
        help="какая сигнатура используется для частотного анализа",
    )
    parser.add_argument("--rarity_q", type=float, default=0.05, help="квантиль редкости (например 0.05)")
    parser.add_argument("--burst_q", type=float, default=0.99, help="квантиль всплеска (например 0.99)")

    # Порог IF
    parser.add_argument("--p", type=float, default=15.0, help="перцентиль для порога score_samples (например 10/15/20)")
    parser.add_argument(
        "--thr_combine",
        choices=["strict", "lenient"],
        default="lenient",
        help="как объединять перцентильный порог и IQR: strict=min (меньше срабатываний), lenient=max (больше)",
    )

    # Как комбинировать IF и частотный слой
    parser.add_argument(
        "--combine_rule",
        choices=["if_only", "freq_only", "and", "or"],
        default="or",
        help="итоговое правило: IF, Frequency, AND или OR",
    )

    parser.add_argument("--random_state", type=int, default=42)

    # Пока фокусируемся на качестве — отчёты/картинки отключены
    parser.add_argument(
        "--save_details",
        action="store_true",
        help="если указано — сохраняем anomaly_results.csv для отладки (Excel/картинки всё равно не строим)",
    )

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # 1) Загрузка + стандартизация
    df_base, X_scaled, _ = load_data(args.input, require_base=True)
    print(f"[INFO] Загружено строк: {len(df_base)}")

    # 2) Обучение Isolation Forest (без учителя)
    model = train_isolation_forest(X_scaled, random_state=args.random_state)

    # 3) Оценки аномальности
    scores = anomaly_scores(model, X_scaled)

    # 4) Порог аномальности (перцентиль + IQR)
    labels, final_thr, p_thr, iqr_thr = detect_anomalies_custom(
        scores,
        p=args.p,
        iqr_k=1.5,
        combine=args.thr_combine,
    )

    # после этой строки у нас появляется таблица, где уже есть и исходные поля трафика, и результат IF (anomaly_score, anomaly_label).
    df_results = build_results_dataframe(df_base, scores, labels)

    # 5) Частотные признаки + окно времени
    if args.freq_key == "dst_port":
        key_cols = ["destination_port"]
    else:
        key_cols = ["destination_port", "protocol"]

    df_results = add_frequency_features(
        df_results,
        window_minutes=args.window,
        key_columns=key_cols,
        rarity_q=args.rarity_q,
        burst_q=args.burst_q,
    )

    # Базовые предикты
    pred_if = (df_results["anomaly_label"] == -1).astype(int) # Перевод -1/1 --> True/False --> 1/0, где 0 - норма, 1 - аномалия
    pred_freq = (df_results["rarity_flag"] | df_results["burst_flag"]).astype(int) # Логическое или между arity_flag и burst_flag --> 1/0, где 0 - норма, 1 - аномалия

    # Итог по выбранному правилу
    if args.combine_rule == "if_only":
        pred_final = pred_if
    elif args.combine_rule == "freq_only":
        pred_final = pred_freq
    elif args.combine_rule == "and":
        pred_final = (pred_if & pred_freq).astype(int)
    else:
        pred_final = (pred_if | pred_freq).astype(int)

    df_results["final_anomaly"] = pred_final # Финальная метка

    # 6) Метрики качества по label (если есть)
    if "label" in df_results.columns:
        y_true = df_results["label"].astype(int).values

        m_if = compute_metrics(y_true, pred_if.values)
        m_if["method"] = "isolation_forest_only"

        m_freq = compute_metrics(y_true, pred_freq.values)
        m_freq["method"] = f"frequency_only({args.freq_key})"

        m_final = compute_metrics(y_true, pred_final.values)
        m_final["method"] = f"final({args.combine_rule})"

        metrics_df = pd.DataFrame([m_if, m_freq, m_final])
        metrics_path = os.path.join(args.output, "metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)

        print("[INFO] Метрики качества сохранены:", metrics_path)
        print("[INFO] IF-only:", {k: v for k, v in m_if.items() if k != "method"})
        print("[INFO] Freq-only:", {k: v for k, v in m_freq.items() if k != "method"})
        print("[INFO] Final:", {k: v for k, v in m_final.items() if k != "method"})

    # 7) Детальные результаты — только если явно попросили
    if args.save_details:
        csv_path = os.path.join(args.output, "anomaly_results.csv")
        df_results.to_csv(csv_path, index=False)
        print(f"[INFO] CSV сохранён: {csv_path}")

    print("[OK] Готово.")


if __name__ == "__main__":
    main()
