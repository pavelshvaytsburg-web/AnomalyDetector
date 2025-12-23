import math
import numpy as np
import pandas as pd


def add_frequency_features(
    df: pd.DataFrame,
    window_minutes: int = 1,
    key_columns=None,
    rarity_q: float = 0.05,
    burst_q: float = 0.99,
) -> pd.DataFrame:
    """
    Добавляет частотные признаки для постобработки детектора аномалий.

    Смысл частотного слоя
    ---------------------
    Этот слой НЕ обучает модель и НЕ использует label. Он добавляет к каждой строке
    два типа "частотных сигналов", которые удобно комбинировать с anomaly_score/IF:

    1) Глобальная редкость (rarity):
       Если некоторый шаблон трафика (например порт назначения или порт+протокол)
       встречается очень редко по всему датасету, это может быть подозрительно.
       Пример: dst_port=9999 встретился 1 раз за весь набор.

    2) Всплеск во времени (burst):
       Если в одном коротком временном окне (например 1 минута) один и тот же шаблон
       повторяется много раз, это похоже на всплеск активности (например DDoS/scan).
       Пример: dst_port=80 в минуте 12:05 встретился 50 раз.

    Почему по умолчанию key_columns = ["destination_port"]
    ------------------------------------------------------
    - source_port обычно эфемерен (часто уникален), и если включить его в ключ,
      то частоты "ломаются": почти все freq_count/window_count становятся равны 1,
      а значит частотный анализ перестаёт быть информативным.

    Параметры
    ---------
    df : pd.DataFrame
        Входной DataFrame. Обязательно должен содержать колонку 'time'
        и колонки, указанные в key_columns (например 'destination_port').

    window_minutes : int
        Размер временного окна в минутах. Используется для группировки по времени:
        time_window = floor(time, window_minutes).
        Пример при window_minutes=1:
        12:05:15 и 12:05:45 попадут в одно окно 12:05:00.

    key_columns : list[str] | None
        Список колонок, определяющих "шаблон" трафика (сигнатуру) для частот.
        Примеры:
        - ["destination_port"]
        - ["destination_port", "protocol"]
        Если None, берём ["destination_port"].

    rarity_q : float
        Квантиль для редкости (нижний хвост распределения freq_count).
        Пример: 0.05 означает, что порог rarity_thr — примерно значение,
        ниже которого лежат "самые редкие 5%" (по freq_count).

    burst_q : float
        Квантиль для всплеска (верхний хвост распределения window_count).
        Пример: 0.99 означает, что порог burst_thr_raw берётся по верхнему 1%.

    Возвращает
    ----------
    pd.DataFrame
        Копию исходного df с добавленными колонками:
        - time_window   : timestamp начала окна (floor по N минутам)
        - freq_count    : сколько раз шаблон встречается во всём датасете
        - window_count  : сколько раз шаблон встречается в текущем time_window
        - rarity_flag   : True, если freq_count <= rarity_thr
        - burst_flag    : True, если window_count >= burst_thr

        Дополнительно (не как колонки, а в out.attrs):
        - rarity_thr    : числовой порог редкости
        - burst_thr     : целочисленный порог всплеска (>=2)
    """

    # Если ключ не задан, используем самый интерпретируемый и устойчивый вариант:
    # частоты по порту назначения.
    if key_columns is None:
        # Важно: source_port НЕ используем — иначе почти все шаблоны уникальны
        # и window_count почти всегда == 1.
        key_columns = ["destination_port"]

    # Копируем, чтобы безопасно добавлять новые колонки и не менять исходный df по ссылке.
    out = df.copy()

    # Для оконной агрегации обязательно нужна колонка времени.
    if "time" not in out.columns:
        raise ValueError("Для оконного частотного анализа нужна колонка 'time'.")

    # Убеждаемся, что time имеет datetime-тип (иначе .dt.floor не сработает).
    if not np.issubdtype(out["time"].dtype, np.datetime64):
        out["time"] = pd.to_datetime(out["time"], errors="coerce")


    # Формируем временное окно: "округляем вниз" timestamp до границы N минут.
    # Пример: при window_minutes=1 время 12:05:45 -> 12:05:00.
    out["time_window"] = out["time"].dt.floor(f"{int(window_minutes)}min")

    # freq_count: глобальная частота шаблона по всему датасету.
    # Для каждой строки считаем, сколько раз её key (например dst_port=80) встречается в целом наборе.
    out["freq_count"] = out.groupby(key_columns)[key_columns[0]].transform("size")

    # window_count: частота шаблона внутри конкретного временного окна.
    # Для каждой строки считаем, сколько раз её key встречается внутри того же time_window.
    out["window_count"] = out.groupby(key_columns + ["time_window"])[key_columns[0]].transform("size")

    # Порог редкости: нижний квантиль распределения freq_count.
    # Пример: rarity_q=0.05 -> "редкими" считаем шаблоны с частотой в нижних ~5%.
    rarity_thr = float(out["freq_count"].quantile(rarity_q))

    # "Сырой" порог всплеска: верхний квантиль распределения window_count.
    # Пример: burst_q=0.99 -> смотрим верхний ~1% значений window_count.
    burst_thr_raw = float(out["window_count"].quantile(burst_q))

    # Защита от вырождения:
    # Если burst_thr_raw получится 1.0 (часто бывает, когда большинство window_count == 1),
    # то условие "window_count >= 1" пометит ВСЕ строки как всплеск.
    # Поэтому:
    # - округляем порог вверх (ceil) до целого,
    # - и делаем минимум 2: "всплеск" = хотя бы 2 события шаблона в одном окне.
    burst_thr = max(2, int(math.ceil(burst_thr_raw)))

    # Флаги, которые потом комбинируются с Isolation Forest:
    # rarity_flag: глобально редкий шаблон (малый freq_count)
    out["rarity_flag"] = out["freq_count"] <= rarity_thr # True or False
    # burst_flag: локальный всплеск во времени (большой window_count)
    out["burst_flag"] = out["window_count"] >= burst_thr # True or False

    # Сохраняем пороги в метаданных DataFrame (удобно для логов/отладки/отчёта).
    out.attrs["rarity_thr"] = rarity_thr
    out.attrs["burst_thr"] = burst_thr

    return out
