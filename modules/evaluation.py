from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def compute_metrics(y_true, y_pred) -> dict:
    """
    Считает базовые метрики качества бинарного классификатора (обычно минимальный набор для ВКР).

    Важно про формат:
    - y_true: истинные метки (ground truth) из датасета
    - y_pred: предсказанные метки детектора
    - В обоих массивах: 1 — аномалия, 0 — норма

    TP — True Positive (истинно-положительные)
    Модель сказала “аномалия” (1), и в реальности это аномалия (1).
    Это “правильно найденные аномалии”.

    FP — False Positive (ложно-положительные)
    Модель сказала “аномалия” (1), но в реальности это норма (0).
    Это ложные срабатывания.

    TN — True Negative (истинно-отрицательные)
    Модель сказала “норма” (0), и в реальности это норма (0).
    Это “правильно распознанные нормальные потоки”.

    FN — False Negative (ложно-отрицательные)
    Модель сказала “норма” (0), но в реальности это аномалия (1).
    Это пропущенные аномалии (самое неприятное для детектора).

    Возвращает:
    - precision, recall, f1
    - false_positive_rate (FPR)
    - tp, fp, tn, fn (элементы матрицы ошибок)
    """

    # Precision (точность): среди всех предсказанных аномалий (y_pred=1),
    # какая доля действительно является аномалиями (y_true=1).
    # Формула: TP / (TP + FP)
    precision = precision_score(y_true, y_pred, zero_division=0)

    # Recall (полнота): среди всех реальных аномалий (y_true=1),
    # какая доля была найдена моделью (y_pred=1).
    # Формула: TP / (TP + FN)
    recall = recall_score(y_true, y_pred, zero_division=0)

    # F1-score: гармоническое среднее precision и recall.
    # Формула: 2 * precision * recall / (precision + recall)
    # zero_division=0: если precision+recall=0 (например модель всё предсказала как 0),
    # возвращаем 0.
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Матрица ошибок (confusion matrix) в фиксированном порядке классов [0, 1]:
    # [[TN, FP],
    #  [FN, TP]]
    #
    # labels=[0,1] важно, чтобы порядок был гарантированным,
    # даже если в данных вдруг отсутствует один из классов.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # False Positive Rate (FPR): доля ложных срабатываний среди всех истинных норм.
    # Формула: FP / (FP + TN)
    # Защита от деления на ноль: если (FP+TN)=0, то FPR=0.
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    # Приводим numpy-типы к обычным Python-типам (float/int),
    # чтобы корректно сохранялось в CSV/JSON и красиво печаталось.
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fpr),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }
