import ipaddress
from typing import Any

import numpy as np
import pandas as pd


def ip_int_to_str(ip_int: Any) -> str:
    """Конвертирует IPv4 из int (как в датасете) в строку вида "A.B.C.D".

    В датасете значения часто приходят как int/float (из-за NaN в pandas), поэтому:
    - приводим к int,
    - при ошибке возвращаем "0.0.0.0" (служебный адрес), чтобы пайплайн не падал.
    """
    try:
        if pd.isna(ip_int):
            return "0.0.0.0"
        return str(ipaddress.IPv4Address(int(ip_int)))
    except Exception:
        return "0.0.0.0"


def ip_scope(ip_str: str) -> str:
    """Классификация IP по "видимости" без GeoIP.

    Это не "страна/город" (для этого нужны GeoIP базы), а топологический контекст,
    который легко обосновать в ВКР:
    - private   : частные диапазоны (RFC1918 и др.)
    - public    : глобально маршрутизируемые адреса
    - loopback  : 127.0.0.0/8
    - special   : multicast / link-local / reserved / 0.0.0.0 и т.п.
    """
    try:
        ip_obj = ipaddress.ip_address(ip_str)
        if ip_obj.is_loopback:
            return "loopback"
        if ip_obj.is_private:
            return "private"
        # В терминах ipaddress is_global ~ "публичный" (глобально маршрутизируемый)
        if getattr(ip_obj, "is_global", False):
            return "public"
        return "special"
    except Exception:
        return "unknown"


def ip_subnet(ip_str: str, prefix: int = 24) -> str:
    """Возвращает строку подсети для IPv4, например "192.168.1.0/24"."""
    try:
        ip_obj = ipaddress.ip_address(ip_str)
        if ip_obj.version != 4:
            return ""
        return str(ipaddress.ip_network(f"{ip_str}/{prefix}", strict=False))
    except Exception:
        return ""


def traffic_direction(src_scope: str, dst_scope: str) -> str:
    """Упрощённая "направленность" трафика по типу адресов.

    Даёт удобный для отчёта признак (без внешних сервисов):
    - internal : private -> private
    - outbound : private -> public
    - inbound  : public  -> private
    - external : public  -> public
    - other    : всё остальное (special/unknown)
    """
    if src_scope == "private" and dst_scope == "private":
        return "internal"
    if src_scope == "private" and dst_scope == "public":
        return "outbound"
    if src_scope == "public" and dst_scope == "private":
        return "inbound"
    if src_scope == "public" and dst_scope == "public":
        return "external"
    return "other"


def _map_unique(series: pd.Series, func) -> pd.Series:
    """Применяет функцию к уникальным значениям Series (быстрее, чем apply по всем строкам)."""
    uniq = pd.unique(series)
    mapping = {v: func(v) for v in uniq}
    return series.map(mapping)


def enrich_geo(df: pd.DataFrame) -> pd.DataFrame:
    """Топологическое ("географическое") обогащение без GeoIP.

    Добавляет контекстные колонки на основе source_ip_int/destination_ip_int:
    - source_ip / destination_ip         : IP в строковом виде
    - source_scope / destination_scope   : private/public/loopback/special
    - source_subnet_24 / destination_subnet_24 : подсеть /24
    - source_subnet_16 / destination_subnet_16 : подсеть /16 (более крупная агрегация)
    - same_subnet_24                     : 1 если src и dst в одной /24, иначе 0
    - traffic_direction                  : internal/outbound/inbound/external/other

    Эти колонки НЕ используются моделью Isolation Forest (если ты их явно не добавишь
    в FEATURE_COLS), но полезны для объяснения результатов и отчёта.
    """
    out = df.copy()

    # Если в данных нет базовых колонок, просто возвращаем как есть.
    if "source_ip_int" not in out.columns or "destination_ip_int" not in out.columns:
        return out

    # 1) int -> строка IP
    out["source_ip"] = _map_unique(out["source_ip_int"], ip_int_to_str)
    out["destination_ip"] = _map_unique(out["destination_ip_int"], ip_int_to_str)

    # 2) scope (private/public/...)
    out["source_scope"] = _map_unique(out["source_ip"], ip_scope)
    out["destination_scope"] = _map_unique(out["destination_ip"], ip_scope)

    # 3) подсети (/24 и /16)
    out["source_subnet_24"] = _map_unique(out["source_ip"], lambda s: ip_subnet(s, 24))
    out["destination_subnet_24"] = _map_unique(out["destination_ip"], lambda s: ip_subnet(s, 24))

    # /16 удобнее для статистики: в синтетических/разреженных данных /24 часто уникальны,
    # а /16 даёт устойчивые группы (например 192.168.0.0/16, 10.0.0.0/16).
    out["source_subnet_16"] = _map_unique(out["source_ip"], lambda s: ip_subnet(s, 16))
    out["destination_subnet_16"] = _map_unique(out["destination_ip"], lambda s: ip_subnet(s, 16))

    # 4) совпадение /24
    out["same_subnet_24"] = (
        (out["source_subnet_24"] != "")
        & (out["destination_subnet_24"] != "")
        & (out["source_subnet_24"] == out["destination_subnet_24"])
    ).astype(int)

    # 5) направление
    out["traffic_direction"] = [
        traffic_direction(s, d) for s, d in zip(out["source_scope"], out["destination_scope"], strict=False)
    ]

    return out


def add_ip_rarity_features(
    df: pd.DataFrame,
    prefix: int = 16,
    rarity_q: float = 0.01,
    side: str = "either",
) -> pd.DataFrame:
    """Минимальная интеграция IP-контекста в детекцию через редкость подсети.

    Идея:
    - строим частоты подсетей (обычно /16) для источника и/или назначения;
    - подсети, которые встречаются крайне редко, помечаем как подозрительные.

    Это НЕ настоящая GeoIP-география (страна/город), а топологический признак:
    "необычный сегмент сети". Он хорошо объясним в ВКР и не требует внешних баз.

    Параметры:
    - prefix: длина префикса подсети (по умолчанию 16). /24 в разреженных данных
      часто даёт почти уникальные значения и плохо подходит для частот.
    - rarity_q: квантиль для порога редкости (например 0.01 = нижние 1%).
    - side: какие подсети учитывать:
        * 'src'    — только источник,
        * 'dst'    — только назначение,
        * 'either' — источник ИЛИ назначение (по умолчанию).

    Добавляет колонки:
    - source_subnet_{prefix}, destination_subnet_{prefix}
    - src_subnet_count, dst_subnet_count
    - ip_rarity_thr
    - ip_rarity_flag (bool)
    """
    if side not in {"src", "dst", "either"}:
        raise ValueError("side должен быть одним из: 'src', 'dst', 'either'")

    out = df.copy()

    # Убедимся, что базовый IP-контекст уже есть. Если нет — обогащаем.
    if "source_ip" not in out.columns or "destination_ip" not in out.columns:
        out = enrich_geo(out)

    src_col = f"source_subnet_{int(prefix)}"
    dst_col = f"destination_subnet_{int(prefix)}"

    # Если нужные подсети ещё не построены (например prefix=20), достраиваем.
    if src_col not in out.columns:
        out[src_col] = _map_unique(out["source_ip"], lambda s: ip_subnet(s, int(prefix)))
    if dst_col not in out.columns:
        out[dst_col] = _map_unique(out["destination_ip"], lambda s: ip_subnet(s, int(prefix)))

    # Частоты подсетей.
    # Важно: порог редкости считаем по РАСПРЕДЕЛЕНИЮ УНИКАЛЬНЫХ ПОДСЕТЕЙ (value_counts),
    # а не по строкам (иначе доминирующая подсеть "утащит" квантиль вверх и всё станет редким).
    src_counts = out[src_col].value_counts(dropna=False)
    dst_counts = out[dst_col].value_counts(dropna=False)

    out["src_subnet_count"] = out[src_col].map(src_counts).fillna(0).astype(int)
    out["dst_subnet_count"] = out[dst_col].map(dst_counts).fillna(0).astype(int)

    # Порог редкости: нижний квантиль по множеству подсетей (а не по строкам).
    # Минимальный разумный порог — 1 (иначе никто не будет "редким").
    thr_src = max(1, int(round(float(np.quantile(src_counts.values, rarity_q)))))
    thr_dst = max(1, int(round(float(np.quantile(dst_counts.values, rarity_q)))))

    if side == "src":
        flag = out["src_subnet_count"] <= thr_src
        thr_info = thr_src
    elif side == "dst":
        flag = out["dst_subnet_count"] <= thr_dst
        thr_info = thr_dst
    else:  # either
        flag = (out["src_subnet_count"] <= thr_src) | (out["dst_subnet_count"] <= thr_dst)
        thr_info = {"src": thr_src, "dst": thr_dst}

    out["ip_rarity_flag"] = flag

    # Сохраняем пороги в attrs (удобно логировать и писать в отчёт).
    out.attrs["ip_rarity_thr"] = thr_info
    out.attrs["ip_rarity_q"] = float(rarity_q)
    out.attrs["ip_rarity_prefix"] = int(prefix)
    out.attrs["ip_rarity_side"] = side

    return out
