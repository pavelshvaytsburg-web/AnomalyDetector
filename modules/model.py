import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def train_isolation_forest(
    X: pd.DataFrame,
    n_estimators: int = 100,
    max_features: float = 1.0,
    random_state: int = None
) -> IsolationForest:
    """
    Обучение модели Isolation Forest в режиме без учителя.
    Мы не задаём contamination, чтобы модель не знала долю аномалий.
    """
    model = IsolationForest(
        n_estimators=n_estimators,
        max_features=max_features,
        random_state=random_state,
        contamination='auto'  # работает только для обучающей части, но predict мы не используем
    )

    model.fit(X)
    print("[INFO] Модель Isolation Forest обучена (unsupervised режим)")
    return model


def detect_anomalies_scores(model: IsolationForest, X: pd.DataFrame):
    """
    Получение оценки аномальности. Меньше = более аномально.
    """
    scores = model.decision_function(X)
    print("[INFO] Оценки аномальности рассчитаны")
    return scores


def detect_anomalies_custom(scores: np.ndarray):
    """
    Собственный механизм определения аномалий.
    Используем два статистических метода:
    1) Перцентильный порог (нижние 5%)
    2) IQR (интерквартильный размах)
    """

    # Перцентильный порог — нижние 5%
    p_threshold = np.percentile(scores, 5)

    # IQR порог
    Q1 = np.percentile(scores, 25)
    Q3 = np.percentile(scores, 75)
    IQR = Q3 - Q1
    iqr_threshold = Q1 - 1.5 * IQR

    # Итоговый порог — комбинируем оба
    final_threshold = min(p_threshold, iqr_threshold)

    # формируем метки аномалий
    labels = np.where(scores < final_threshold, -1, 1)

    print("[INFO] Аномалии определены через перцентиль и IQR")
    print(f"       Порог (перцентиль): {p_threshold}")
    print(f"       Порог (IQR): {iqr_threshold}")
    print(f"       Итоговый порог: {final_threshold}")

    return labels, final_threshold, p_threshold, iqr_threshold


def build_results_dataframe(df_features: pd.DataFrame, scores, labels):
    """
    Формирование итоговой таблицы результатов.
    """
    result = df_features.copy()
    result["anomaly_score"] = scores
    result["anomaly_label"] = labels
    return result
