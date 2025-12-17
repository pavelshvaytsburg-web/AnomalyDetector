import pandas as pd
import ipaddress

def int_to_ipv4(ip_int):
    """3232235777 -> '192.168.1.1'"""
    return str(ipaddress.IPv4Address(int(ip_int)))

def ipv4_to_subnet(ip_str, prefix_len):
    """'192.168.1.123', 24 -> '192.168.1.0/24'"""
    net = ipaddress.IPv4Network(f"{ip_str}/{prefix_len}", strict=False)
    return str(net)

# 1) читаем датасет
df = pd.read_csv("network_traffic.csv")

# 2) покажем первые 5 строк: int -> IPv4 -> /24
n = 5
for i in range(n):
    src_int = df.loc[i, "source_ip_int"]
    dst_int = df.loc[i, "destination_ip_int"]

    src_ip = int_to_ipv4(src_int)
    dst_ip = int_to_ipv4(dst_int)

    src_net24 = ipv4_to_subnet(src_ip, 24)
    dst_net24 = ipv4_to_subnet(dst_ip, 24)

    print(f"Row {i}: src={src_int} -> {src_ip} -> {src_net24} | dst={dst_int} -> {dst_ip} -> {dst_net24}")

# 3) если нужно добавить колонки в таблицу (тоже без lambda)
src_ip_list = []
dst_ip_list = []
src_net24_list = []
dst_net24_list = []

for i in range(len(df)):
    src_ip = int_to_ipv4(df.loc[i, "source_ip_int"])
    dst_ip = int_to_ipv4(df.loc[i, "destination_ip_int"])

    src_ip_list.append(src_ip)
    dst_ip_list.append(dst_ip)
    src_net24_list.append(ipv4_to_subnet(src_ip, 24))
    dst_net24_list.append(ipv4_to_subnet(dst_ip, 24))

df["source_ip"] = src_ip_list
df["destination_ip"] = dst_ip_list
df["source_net24"] = src_net24_list
df["destination_net24"] = dst_net24_list
