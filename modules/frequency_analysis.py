import pandas as pd

def frequency_analysis(df_results: pd.DataFrame, key_columns=None):
    """
    Добавляет в таблицу две новые колонки:
    1) freq_count — сколько раз встречается такой тип потока
    2) rarity_flag — True, если запись редкая и встречается редко
    """

    if key_columns is None:
        # Определяем по портам и протоколу (базовое поведение)
        key_columns = ["source_port", "destination_port", "protocol"]

    # Считаем частоту по ключевым столбцам
    freq_series = df_results.groupby(key_columns).size()

    # Добавляем частоту в таблицу
    df_results["freq_count"] = df_results[key_columns].apply(
        lambda row: freq_series[tuple(row)], axis=1
    )

    # Теперь определяем редкость: если частота <= 2 (можно менять)
    df_results["rarity_flag"] = df_results["freq_count"] <= 2

    return df_results
