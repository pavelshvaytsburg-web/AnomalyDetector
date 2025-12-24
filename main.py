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
from modules.geo import enrich_geo, add_ip_rarity_features


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

    # Минимальное использование IP-контекста в детекции (не GeoIP, а редкость сетевых сегментов)
    parser.add_argument(
        "--use_ip_rarity",
        action="store_true",
        help=(
            "если указано — добавляем простой IP-сигнал 'редкая подсеть' и можем комбинировать его с итогом. "
            "Полезно как дополнительный контекстный слой (без внешних GeoIP баз)."
        ),
    )
    parser.add_argument("--ip_prefix", type=int, default=16, help="префикс подсети для IP-редкости (по умолчанию /16)")
    parser.add_argument("--ip_rarity_q", type=float, default=0.01, help="квантиль редкости подсети (например 0.01)")
    parser.add_argument(
        "--ip_side",
        choices=["src", "dst", "either"],
        default="either",
        help="какие подсети учитывать для редкости: src/dst/either",
    )
    parser.add_argument(
        "--ip_combine",
        choices=["or", "and"],
        default="or",
        help=(
            "как комбинировать IP-сигнал с итоговым предиктом: "
            "or — повышает recall, and — повышает precision (может снизить recall)"
        ),
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

    # 5.1) "География" / топология узлов (без GeoIP баз и внешних сервисов)
    # Добавляем IP в строковом виде, тип адресов (private/public/...) и подсети.
    # На качество детекции это не влияет, если ты не добавляешь эти колонки в FEATURE_COLS,
    # но это полезно для интерпретации результатов в отчёте.
    df_results = enrich_geo(df_results)

    # 5.2) (опционально) IP-сигнал "редкая подсеть".
    # Важное: это не "обучение по IP", а простой частотный критерий по сетевым сегментам.
    # Включается только флагом --use_ip_rarity.
    if args.use_ip_rarity:
        df_results = add_ip_rarity_features(
            df_results,
            prefix=args.ip_prefix,
            rarity_q=args.ip_rarity_q,
            side=args.ip_side,
        )

    # Базовые предикты
    pred_if = (df_results["anomaly_label"] == -1).astype(int)
    pred_freq = (df_results["rarity_flag"] | df_results["burst_flag"]).astype(int)

    # IP-предикт (если включён): 1 = редкая подсеть, 0 = обычная
    pred_ip = None
    if args.use_ip_rarity:
        pred_ip = df_results["ip_rarity_flag"].astype(int)

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

    # При необходимости учитываем IP-сигнал
    if args.use_ip_rarity and pred_ip is not None:
        if args.ip_combine == "and":
            pred_final_ip = (pred_final & pred_ip).astype(int)
        else:  # "or"
            pred_final_ip = (pred_final | pred_ip).astype(int)
        df_results["final_anomaly"] = pred_final_ip
        pred_final = pred_final_ip

    # 6) Метрики качества по label (если есть)
    if "label" in df_results.columns:
        y_true = df_results["label"].astype(int).values

        m_if = compute_metrics(y_true, pred_if.values)
        m_if["method"] = "isolation_forest_only"

        m_freq = compute_metrics(y_true, pred_freq.values)
        m_freq["method"] = f"frequency_only({args.freq_key})"

        m_final = compute_metrics(y_true, pred_final.values)
        m_final["method"] = f"final({args.combine_rule})" + ("+ip" if args.use_ip_rarity else "")

        rows = [m_if, m_freq, m_final]

        # Если IP-сигнал включён, полезно показать его вклад отдельной строкой
        if args.use_ip_rarity and pred_ip is not None:
            m_ip = compute_metrics(y_true, pred_ip.values)
            m_ip["method"] = f"ip_rarity(/ {args.ip_prefix}, q={args.ip_rarity_q}, side={args.ip_side})"
            rows.insert(2, m_ip)

        metrics_df = pd.DataFrame(rows)
        metrics_path = os.path.join(args.output, "metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)

        print("[INFO] Метрики качества сохранены:", metrics_path)
        print("[INFO] IF-only:", {k: v for k, v in m_if.items() if k != "method"})
        print("[INFO] Freq-only:", {k: v for k, v in m_freq.items() if k != "method"})
        if args.use_ip_rarity and pred_ip is not None:
            print("[INFO] IP-rarity:", {k: v for k, v in m_ip.items() if k != "method"})
        print("[INFO] Final:", {k: v for k, v in m_final.items() if k != "method"})

    # 7) Детальные результаты — только если явно попросили
    if args.save_details:
        csv_path = os.path.join(args.output, "anomaly_results.csv")
        df_results.to_csv(csv_path, index=False)
        print(f"[INFO] CSV сохранён: {csv_path}")

    print("[OK] Готово.")


if __name__ == "__main__":
    main()
