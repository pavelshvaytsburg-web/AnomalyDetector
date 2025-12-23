import numpy as np
import pandas as pd


def permutation_importance_unsupervised(
    model,
    X_scaled: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Перестановочная важность признаков в безучительском сценарии.

    Идея:
    - считаем базовые score_samples
    - затем по одному признаку перемешиваем колонку и смотрим, насколько изменятся...

    Важно:
    - X_scaled ДОЛЖЕН быть тем же представлением, на котором обучена модель.
      Иначе модель видит "другие" масштабы признаков и оценка искажается.
    """

    rng = np.random.default_rng(random_state)

    base_scores = model.score_samples(X_scaled)
    importances = {}

    for j, name in enumerate(feature_names):
        diffs = []
        for _ in range(n_repeats):
            X_shuffled = X_scaled.copy()
            X_shuffled[:, j] = rng.permutation(X_shuffled[:, j])

            new_scores = model.score_samples(X_shuffled)
            diffs.append(float(np.mean(np.abs(base_scores - new_scores))))

        importances[name] = float(np.mean(diffs))

    return (
        pd.DataFrame({"feature": list(importances.keys()), "importance": list(importances.values())})
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )
