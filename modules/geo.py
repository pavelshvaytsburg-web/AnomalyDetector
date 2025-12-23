import ipaddress


def ip_int_to_str(ip_int: int) -> str:
    """Конвертация IPv4 из int (как в датасете) в строку."""
    try:
        return str(ipaddress.IPv4Address(int(ip_int)))
    except Exception:
        return "0.0.0.0"


def ip_scope(ip_str: str) -> str:
    """Классификация адреса на уровне видимости.

    Используем стандартную логику ipaddress:
    - is_private покрывает RFC1918 (10/8, 172.16/12, 192.168/16) и др. частные диапазоны;
    - is_loopback — 127.0.0.0/8.

    Для отчёта ВКР это более чем достаточно (без GeoIP баз и внешних сервисов).
    """
    try:
        ip_obj = ipaddress.ip_address(ip_str)
        if ip_obj.is_loopback:
            return "loopback"
        if ip_obj.is_private:
            return "private"
        # is_global == "публичный" в терминах ipaddress
        if getattr(ip_obj, "is_global", False):
            return "public"
        return "special"  # link-local, multicast, reserved, etc.
    except Exception:
        return "unknown"


def ip_subnet(ip_str: str, prefix: int = 24) -> str:
    """Возвращает строку подсети для IPv4, например 192.168.1.0/24."""
    try:
        ip_obj = ipaddress.ip_address(ip_str)
        if ip_obj.version != 4:
            return ""
        net = ipaddress.ip_network(f"{ip_str}/{prefix}", strict=False)
        return str(net)
    except Exception:
        return ""


def enrich_geo(df):
    """Топологическое/"географическое" обогащение без внешних сервисов.

    Что добавляем:
    - source_ip / destination_ip: IPv4 в строковом виде
    - *_scope: private/public/loopback/special
    - *_subnet_24: подсеть /24
    - *_subnet_16: подсеть /16
    - same_subnet_24: флаг совпадения /24 (часто полезно для ИБ-аналитики)
    """
    out = df.copy()

    if "source_ip_int" in out.columns:
        out["source_ip"] = out["source_ip_int"].apply(ip_int_to_str)
        out["source_scope"] = out["source_ip"].apply(ip_scope)
        out["source_subnet_24"] = out["source_ip"].apply(lambda s: ip_subnet(s, 24))
        out["source_subnet_16"] = out["source_ip"].apply(lambda s: ip_subnet(s, 16))

    if "destination_ip_int" in out.columns:
        out["destination_ip"] = out["destination_ip_int"].apply(ip_int_to_str)
        out["destination_scope"] = out["destination_ip"].apply(ip_scope)
        out["destination_subnet_24"] = out["destination_ip"].apply(lambda s: ip_subnet(s, 24))
        out["destination_subnet_16"] = out["destination_ip"].apply(lambda s: ip_subnet(s, 16))

    # Совпадение /24 (если обе подсети определились)
    if "source_subnet_24" in out.columns and "destination_subnet_24" in out.columns:
        out["same_subnet_24"] = (
            (out["source_subnet_24"] != "")
            & (out["destination_subnet_24"] != "")
            & (out["source_subnet_24"] == out["destination_subnet_24"])
        ).astype(int)

    return out
