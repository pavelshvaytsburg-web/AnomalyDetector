import pandas as pd
from sklearn.preprocessing import StandardScaler


# Единый источник истины по признакам модели
FEATURE_COLS = [
    "source_port",
    "destination_port",
    "protocol",
    "duration",
    "packet_count",
    "bytes_sent",
    "bytes_received",
    "bytes_per_packet",
]

# Базовые поля для временных окон и топологии ("географии")
BASE_COLS = ["time", "source_ip_int", "destination_ip_int"]


def load_data(path: str, require_base: bool = True):
    """
    Загружает CSV и готовит данные для модели.

    Важно:
    - Дубликаты НЕ удаляем, т.к. частота повторов используется в частотном анализе.
    - Приводим time к datetime, признаки к numeric.
    """

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Ошибка при чтении файла {path}: {e}")

    # Проверяем схему
    missing_features = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_features:
        raise ValueError(f"В датасете отсутствуют столбцы признаков: {missing_features}")

    if require_base:
        missing_base = [c for c in BASE_COLS if c not in df.columns]
        if missing_base:
            raise ValueError(f"В датасете отсутствуют базовые колонки: {missing_base}")

    # time для оконного анализа
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # Признаки -> числа (в CSV иногда бывают строки/пустые значения)
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # IP int -> числа (если есть)
    for col in ["source_ip_int", "destination_ip_int"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # label (если есть) приводим к 0/1 int: нужна только для оценки качества, НЕ для обучения
    if "label" in df.columns:
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)

    # Удаляем строки, где нет ключевых полей (НО НЕ drop_duplicates)
    drop_subset = [c for c in (["time"] if "time" in df.columns else []) + FEATURE_COLS if c in df.columns]
    df = df.dropna(subset=drop_subset).reset_index(drop=True)

    # На выходе держим базовые + признаки (для отчёта) + label (если присутствует)
    extra_cols = ["label"] if "label" in df.columns else []
    if require_base:
        df_out = df[BASE_COLS + FEATURE_COLS + extra_cols].copy()
    else:
        df_out = df[FEATURE_COLS + extra_cols].copy()

# Стандартизируем признаки перед обучением модели:
# StandardScaler приводит каждый столбец к виду z = (x - mean) / std,
# т.е. среднее ≈ 0 и стандартное отклонение ≈ 1.
# Это нужно, чтобы признаки с большими масштабами (например bytes) не "доминировали"
# над признаками с малыми значениями (например protocol/ports) при обучении Isolation Forest.

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_out[FEATURE_COLS].values)

    return df_out, X_scaled, scaler
