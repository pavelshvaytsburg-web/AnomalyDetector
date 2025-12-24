import argparse
import os

import numpy as np
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
from modules.visualization import generate_all_plots


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
        help="какой ключ использовать для частот: dst_port или dst_port+protocol",
    )
    parser.add_argument("--rarity_q", type=float, default=0.05, help="квантиль редкости (ниже — редкое)")
    parser.add_argument("--burst_q", type=float, default=0.99, help="квантиль всплеска (выше — всплеск)")

    # Порог по Isolation Forest
    parser.add_argument("--p", type=float, default=0.02, help="перцентиль для порога аномальности (нижний хвост)")
    parser.add_argument(
        "--thr_combine",
        choices=["strict", "lenient"],
        default="strict",
        help="как комбинировать p-порог и IQR-порог (strict=min, lenient=max)",
    )

    # Итоговое правило объединения IF и Frequency
    parser.add_argument(
        "--combine_rule",
        choices=["if_only", "freq_only", "and", "or"],
        default="or",
        help="итоговое правило: IF, Frequency, AND или OR",
    )

    parser.add_argument("--random_state", type=int, default=42)

    # Визуализация одним флагом
    parser.add_argument(
        "--plots",
        action="store_true",
        help="если указано — сохраняем набор графиков в папку <output>/plots",
    )

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # 1) Загрузка + стандартизация
    df_base, X_scaled, _ = load_data(args.input, require_base=True)
    print(f"[INFO] Загружено строк: {len(df_base)}")

    def _metrics_to_text_table(metrics_df: pd.DataFrame) -> str:
        # Чуть аккуратнее печать float-метрик
        dfp = metrics_df.copy()
        for col in dfp.columns:
            if col == "method":
                continue
            if np.issubdtype(dfp[col].dtype, np.floating):
                dfp[col] = dfp[col].round(4)
        return dfp.to_string(index=False)

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

    df_results["final_anomaly"] = pred_final


    # 6) Метрики качества по label (если есть)
    if "label" in df_results.columns:
        y_true = df_results["label"].astype(int).values

        m_if = compute_metrics(y_true, pred_if.values)
        m_if["method"] = "isolation_forest_only"

        m_freq = compute_metrics(y_true, pred_freq.values)
        m_freq["method"] = f"frequency_only({args.freq_key})"

        m_final = compute_metrics(y_true, pred_final.values)
        m_final["method"] = f"final({args.combine_rule})"

        rows = [m_if, m_freq, m_final]

        metrics_df = pd.DataFrame(rows)

        # Метрики сохраняем в читабельном виде (таблица), без CSV
        metrics_txt_path = os.path.join(args.output, "metrics.txt")
        table = _metrics_to_text_table(metrics_df)
        with open(metrics_txt_path, "w", encoding="utf-8") as f:
            f.write(f"Input: {args.input}\n")
            f.write(
                "Config: "
                f"combine_rule={args.combine_rule}, p={args.p}, thr_combine={args.thr_combine}, "
                f"window={args.window}, freq_key={args.freq_key}, rarity_q={args.rarity_q}, burst_q={args.burst_q}\n"
            )
            f.write(f"Thresholds: p_thr={p_thr:.6f}, iqr_thr={iqr_thr:.6f}, final_thr={final_thr:.6f}\n\n")
            f.write(table + "\n")
            
        print("[INFO] Метрики качества сохранены:", metrics_txt_path)
        print(table)

        print("[INFO] IF-only:", {k: v for k, v in m_if.items() if k != "method"})
        print("[INFO] Freq-only:", {k: v for k, v in m_freq.items() if k != "method"})
        print("[INFO] Final:", {k: v for k, v in m_final.items() if k != "method"})


    # 8) Визуализация
    if args.plots:
        plots_dir = generate_all_plots(
            df_results,
            output_dir=args.output,
            final_thr=final_thr,
            p_thr=p_thr,
            iqr_thr=iqr_thr,
            window_minutes=args.window,  # чтобы таймлайн был согласован с частотным окном
        )
        print(f"[INFO] Графики сохранены в: {plots_dir}")

if __name__ == "__main__":
    main()
