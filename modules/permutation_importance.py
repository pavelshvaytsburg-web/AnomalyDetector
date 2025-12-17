import numpy as np
import pandas as pd

def permutation_importance_unsupervised(model, X: pd.DataFrame, n_repeats=5):
    base_scores = model.decision_function(X)
    importances = {}

    for col in X.columns:
        diffs = []
        for _ in range(n_repeats):
            X_shuffled = X.copy()
            X_shuffled[col] = np.random.permutation(X_shuffled[col])

            new_scores = model.decision_function(X_shuffled)
            diff = np.mean(np.abs(base_scores - new_scores))
            diffs.append(diff)

        importances[col] = np.mean(diffs)

    return pd.DataFrame({
        "feature": list(importances.keys()),
        "importance": list(importances.values())
    }).sort_values(by="importance", ascending=False)
