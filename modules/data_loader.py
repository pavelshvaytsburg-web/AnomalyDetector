import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(path: str):
    """
    Загрузка network_traffic.csv.
    Берём только нужные числовые признаки.
    CSV должен быть в UTF-8 с разделителем ','.
    """

    # Загружаем CSV
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Ошибка при чтении файла {path}: {e}")

    # Признаки, которые используем в модели
    required_cols = [
        "source_port",
        "destination_port",
        "protocol",
        "duration",
        "packet_count",
        "bytes_sent",
        "bytes_received",
        "bytes_per_packet"
    ]

    # Проверяем, что в CSV есть все нужные колонки
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"В датасете отсутствуют столбцы: {missing}")

    # Берём только нужные признаки
    df_features = df[required_cols].copy()

    # Очистка данных
    df_features = df_features.dropna().drop_duplicates()

    # Стандартизируем
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)

    return df_features, X_scaled, scaler
