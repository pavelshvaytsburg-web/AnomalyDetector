import numpy as np
from sklearn.ensemble import IsolationForest


def train_isolation_forest(
    X_scaled, # Матрица размером N x 8 (N строк, 8 признаков), ужестандартизированная (StandartScaler)
    n_estimators: int = 100, # количество деревьев в ансамбле. Больше деревьев -> стабильнее оценки anomaly score, но дольше обучение.
    max_features: float = 1.0, # Использовать все признаки
    random_state: int | None = 42, # Фиксация случайности для воспроизводимости
    contamination: str = "auto", # Не используем метод model.predict() -> contamination не играет никакой роли
) -> IsolationForest:
    """
    Обучает Isolation Forest в безучительском режиме на стандартизированных признаках.

    Идея алгоритма:
    - строит ансамбль случайных деревьев (isolation trees),
    - точки, которые "изолируются" за меньшее число разбиений, считаются более аномальными.

    В этом проекте мы:
    - НЕ используем model.predict;
    - НЕ используем нулевой порог decision_function;
    - строим свой порог по распределению score_samples.

    Поэтому contamination можно оставить "auto" как значение по умолчанию.
    """

    model = IsolationForest(
        n_estimators=n_estimators,
        max_features=max_features,
        random_state=random_state,
        contamination=contamination,
    )

    model.fit(X_scaled)
    print("[INFO] Isolation Forest обучен")
    return model


def anomaly_scores(model: IsolationForest, X_scaled) -> np.ndarray:
    """Возвращает score_samples: чем меньше, тем более аномально."""
    scores = model.score_samples(X_scaled) # массив длинны N (N - кол-во строк в датафрейме)
    print("[INFO] anomaly scores (score_samples) рассчитаны")
    return scores


def detect_anomalies_custom(
    scores: np.ndarray,
    p: float = 5.0,
    iqr_k: float = 1.5,
    combine: str = "lenient",
):
    """Собственный механизм определения аномалий.

    Используем два статистических критерия:
    1) Перцентильный порог (нижние p%)
    2) IQR (Q1 - k*IQR)

    combine управляет тем, как объединять два порога:
    - "strict"  : берём более строгий (min). Это уменьшает число срабатываний,
                  обычно повышает precision, но снижает recall.
    - "lenient" : берём менее строгий (max). Это увеличивает число срабатываний,
                  обычно повышает recall, но может добавить FP.

    Важно: для score_samples "более аномально" означает "меньше".
    """

    p_threshold = float(np.percentile(scores, p)) # вычисляет значение, ниже которого лежит p% элементов массива scores.

    q1 = float(np.percentile(scores, 25)) # q1 — это “нижний квартиль”: значение, ниже которого лежит 25% score.
    q3 = float(np.percentile(scores, 75)) # q3 — верхний квартиль: ниже него лежит 75% score.
    iqr = q3 - q1 # qr показывает “типичный разброс” score без крайних хвостов. Если iqr маленький — распределение плотное, если большой — разброс широкий
    """Смысл
    мы берём “типичную нижнюю границу” q1
    и уходим вниз на iqr_k * iqr
    получаем границу, ниже которой значения считаются статистически необычно низкими (выбросами)
    iqr_k - это коэффициент “насколько далеко отойти вниз от q1
    """
    iqr_threshold = q1 - iqr_k * iqr

    if combine == "strict": # более низкий порог --> меньше аномалий и меньше ложносрабатываний
        final_threshold = min(p_threshold, iqr_threshold)
    elif combine == "lenient": # порог выше --> больше аномалий и больше ложносрабатываний
        final_threshold = max(p_threshold, iqr_threshold)
    else:
        raise ValueError("combine должен быть 'strict' или 'lenient'")

# если scores[i] < final_threshold → labels[i] = -1 (аномалия)
# иначе → labels[i] = 1 (норма)
    labels = np.where(scores < final_threshold, -1, 1) 

    print("[INFO] Аномалии определены через перцентиль и IQR")
    print(f"       Порог (p={p}%): {p_threshold}")
    print(f"       Порог (IQR): {iqr_threshold}")
    print(f"       Итоговый порог ({combine}): {final_threshold}")

# labels — метка по каждому потоку (-1/1)
# final_threshold — итоговый порог, который реально использовался
# p_threshold — порог по перцентилю (для отчётности/понимания)
# iqr_threshold — порог по IQR (для отчётности/понимания)
    return labels, final_threshold, p_threshold, iqr_threshold


def build_results_dataframe(df_features, scores, labels):
    out = df_features.copy()
    out["anomaly_score"] = scores
    out["anomaly_label"] = labels
    return out
