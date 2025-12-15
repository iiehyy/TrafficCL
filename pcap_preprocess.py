import os
import json
import pandas as pd
import ast
import json
import math
import os
import subprocess
from audioop import error
from multiprocessing import Pool, Manager
import numpy as np
import pandas as pd
from io import StringIO  # 使用Python标准库的StringIO
from collections import defaultdict
import ipaddress
Base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
def geohash_encode(lat, lng, n):
    # 二进制编码
    lat_num = (5 * n) // 2
    lng_num = 5 * n - lat_num
    lng_str, lat_str = '', ''
    longitudes, latitudes = [[-180, 180]], [[-90, 90]]
    for _ in range(lng_num):
        left, right = longitudes[-1]
        if lng < (left + right) / 2:
            longitudes.append([left, (left + right) / 2])
            lng_str += '0'
        else:
            longitudes.append([(left + right) / 2, right])
            lng_str += '1'
    for _ in range(lat_num):
        left, right = latitudes[-1]
        if lat < (left + right) / 2:
            latitudes.append([left, (left + right) / 2])
            lat_str += '0'
        else:
            latitudes.append([(left + right) / 2, right])
            lat_str += '1'

    # 交叉合并
    str_bin = ''
    for i in range(5 * n):
        if i % 2 == 0:
            str_bin += lng_str[i // 2]
        else:
            str_bin += lat_str[i // 2]

    # Base32编码
    code = ''
    for i in range(n):
        code += Base32[int(str_bin[i * 5: i * 5 + 5], 2)]

    return code
def write_error(file_path,errorlist):
    errorpath = "error_records.csv"
    if file_path not in errorlist:
        error_info = {
            'capture_file':file_path
        }
        error_df = pd.DataFrame([error_info])
        error_df.to_csv(errorpath, mode='a', header=False, index=False, encoding="utf-8")
def ip_to_int(ip):
    parts = ip.split('.')
    return (int(parts[0]) << 24) + (int(parts[1]) << 16) + (int(parts[2]) << 8) + int(parts[3])
def iqr_based_smoothing(data, window_size=3, iqr_factor=1.5):
    """
    使用IQR方法过滤异常值后进行分段均值平滑

    参数:
        data: 原始数据列表/数组
        window_size: 分段大小(默认3)
        iqr_factor: IQR倍数(默认1.5)

    返回:
        (filtered_data, segment_means, final_mean)
        filtered_data: 过滤后的数据
        segment_means: 分段均值列表
        final_mean: 最终均值
    """
    # 计算IQR阈值
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    lower_bound = max(0,(q25 - iqr_factor * iqr))
    upper_bound = q75 + iqr_factor * iqr

    # 过滤异常值
    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
    filtered_data=filtered_data
    #error_data = [x for x in data if lower_bound > x  or x > upper_bound]
    # 分段均值计算
    segment_means = []
    for i in range(0, len(filtered_data), window_size):
        segment = filtered_data[i:i + window_size]
        segment_means.append(float(np.mean(segment)))

    # 计算最终均值
    final_mean = float(np.mean(segment_means))

    return final_mean
def count_positive_negative_diff(lst):
    positive = 0
    negative = 0
    for num in lst:
        if isinstance(num, (int, float)):  # 确保元素是数值类型
            if num >= 0:
                positive += 1
            elif num < 0:
                negative += 1
    return positive/negative
def ip_to_cidr24_and_int(ip_str):
    """将IP地址转换为CIDR/24网段和整数形式

    Args:
        ip_str (str): 输入的IP地址，如 '192.168.1.100'

    Returns:
        tuple: (cidr24_str, cidr24_int)
            cidr24_str: CIDR/24网段，如 '192.168.1.0/24'
            cidr24_int: 网段的整数表示，如 3232235776
    """
    # 解析IP地址
    ip = ipaddress.IPv4Address(ip_str)

    # 转换为CIDR/24网段
    network = ipaddress.IPv4Network(f"{ip_str}/24", strict=False)
    cidr24_str = str(network)

    # 计算CIDR/24网段的整数值
    cidr24_int = int(network.network_address)

    return cidr24_str, cidr24_int
def get_mean(data):
    return float(np.mean(data))
def get_mean_burst(data):
    # 将列表转为NumPy数组，并对所有元素取绝对值
    abs_data = np.abs(data)
    # 计算绝对值数组的均值，并转为Python浮点数
    return float(np.mean(abs_data))
def get_payload_featuree(payloads):
    zero_count = payloads.count(0)  # 零值包数量 = 9
    total_packets = len(payloads)  # 总包数 = 18
    tcp_zero_payload_ratio = zero_count / total_packets
    non_zero_payloads = [x for x in payloads if x != 0]  # 过滤零值
    non_zero_count = len(non_zero_payloads)  # 非零包数 = 9
    if non_zero_count==0:
        return -1,-1
    sum_payload = sum(non_zero_payloads)  # 非零负载总和 = 6254
    tcp_effective_payload_mean = sum_payload / non_zero_count  # 6254/9 ≈ 694.89 字节
    return tcp_zero_payload_ratio,tcp_effective_payload_mean
def extract_location(address):
    """
    从地址字符串中提取国家、城市、区
    地址格式规律：通常为“省/直辖市 市/直辖市 区 详细地址”（国内地址默认国家为中国）
    返回：(Country, City, District)
    """
    # 处理空地址或无效地址
    if pd.isna(address) or str(address).strip() == "":
        return ("未知", "未知", "未知")

    # 统一处理：转换为字符串，去除首尾空格，按空格分割并过滤空字符串
    address_str = str(address).strip()
    parts = [part.strip() for part in address_str.split(" ") if part.strip() != ""]
    part_count = len(parts)

    # 国家默认“中国”（根据地址特征，未出现其他国家，可根据实际数据扩展）
    country = "中国"

    # 提取城市（City）和区（District）
    city = "未知"
    district = "未知"

    if part_count >= 2:
        # 情况1：省级为直辖市（如“北京市 北京市...”“天津市 天津市...”）
        if parts[0] in ["北京市", "上海市", "天津市", "重庆市"]:
            city = parts[0]  # 直辖市名称即为城市名
            if part_count >= 3:
                district = parts[2]  # 第三部分为区
        else:
            # 情况2：省级为省（如“河北省 廊坊市...”“山东省 德州市...”）
            city = parts[1]  # 第二部分为城市名
            if part_count >= 3:
                district = parts[2]  # 第三部分为区
    elif part_count == 1:
        # 仅1部分地址（如“大兴区”），无法明确城市，仅尝试提取区
        if any(key in parts[0] for key in ["区", "县", "市"]):
            district = parts[0]

    return (country, city, district)
def is_private_ip(ip):
    try:
        ip_obj = ipaddress.ip_address(ip)
        return ip_obj.is_private
    except ValueError:
        # 如果传入的IP格式不正确
        return False
def tcp_flow_is_validate(pcap_file):
    tshark_path = r"C:\Program Files\Wireshark\tshark.exe"
    cmd = [
        tshark_path,
        "-r", pcap_file,
        "-T", "fields",
        "-e", "frame.time_epoch",
        "-e", "frame.time_relative",
        "-e", "tcp.analysis.ack_rtt",
        "-e", "tcp.analysis.initial_rtt",
        "-e", "tcp.stream",
        "-e", "tcp.seq",
        "-e", "tcp.ack",
        "-e", "tcp.flags",
        "-e", "tcp.len",
        "-e", "tcp.window_size",
        "-e", "ip.src",
        "-e", "ip.dst",
        "-e", "tcp.srcport",
        "-e", "tcp.dstport",
        "-e", "tcp.analysis.retransmission",
        "-e", "tcp.analysis.duplicate_ack",
        "-e", "tcp.analysis.fast_retransmission",
        "-E", "header=y",
        "-E", "separator=,",
        "-E", "occurrence=f"
    ]

    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        output = output.decode('utf-8')
    except subprocess.CalledProcessError as e:
        print(f"tshark执行失败: {e.output.decode('utf-8')}")
        return pd.DataFrame()
    # 1. 新增：先判断tshark输出是否为空，避免生成空DataFrame
    if not output.strip():
        print(f"⚠️ {pcap_file} tshark未提取到任何数据")
        return pd.DataFrame()
    # 读取原始数据并处理缺失值
    df = pd.read_csv(
        StringIO(output),
        dtype={'frame.time_epoch': float, 'frame.time_relative': float},
        na_values=[""]
    )
    if df.empty:
        print(f"⚠️ {pcap_file} 无有效TCP数据包")
        return pd.DataFrame()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df['tcp.stream'] = df['tcp.stream'].astype(int)
    df['tcp.flags'] = df['tcp.flags'].apply(lambda x: int(x, 16) if pd.notnull(x) else 0)

    # --- 新增：分析流状态 ---
    # --- 分析流状态 ---
    stream_validation = defaultdict(dict)
    # 3. 新增：判断分组后是否有数据（避免stream_validation为空）
    stream_groups = df.groupby('tcp.stream')
    if not stream_groups.groups:
        print(f"⚠️ {pcap_file} 无有效TCP流")
        return pd.DataFrame()

    for stream_id, group in df.groupby('tcp.stream'):
        stats = {
            # 握手阶段
            'syn_sent': False,  # 客户端发送SYN
            'syn_ack_received': False,  # 服务端回复SYN-ACK
            'handshake_ack': False,  # 客户端完成握手ACK
            # 关闭阶段
            'fin_initiated': False,  # 任意一端发送FIN
            'fin_responded': False,  # 对端回复FIN-ACK
            'closure_ack': False,  # FIN发起方确认最终ACK
            # 异常状态
            'rst_received': False,  # 连接被重置
            # 统计信息
            'retransmission_count': 0,
            'duplicate_ack_count': 0,
            'fast_retransmit_count': 0,
            'bytes_transferred': 0,
            'has_payload': False
        }

        # 遍历数据包判断状态
        for _, row in group.iterrows():
            flags = row['tcp.flags']

            # 统计重传
            stats['retransmission_count'] += 1 if pd.notnull(row['tcp.analysis.retransmission']) else 0
            stats['duplicate_ack_count'] += 1 if pd.notnull(row['tcp.analysis.duplicate_ack']) else 0
            stats['fast_retransmit_count'] += 1 if pd.notnull(row['tcp.analysis.fast_retransmission']) else 0

            # 统计有效载荷
            stats['bytes_transferred'] += int(row['tcp.len']) if pd.notnull(row['tcp.len']) else 0

            # 检测握手阶段
            if flags & 0x02:  # SYN标志
                if not (flags & 0x10):  # 纯SYN包 (非SYN-ACK)
                    stats['syn_sent'] = True
            if flags & 0x12 == 0x12:  # SYN和ACK同时置位
                stats['syn_ack_received'] = True
            if stats['syn_sent'] and stats['syn_ack_received']:
                if flags & 0x10 and not (flags & 0x02):  # 纯ACK完成握手
                    stats['handshake_ack'] = True

            # 检测关闭阶段
            if flags & 0x01:  # FIN标志
                stats['fin_initiated'] = True
            if stats['fin_initiated'] and flags & 0x10:  # 对端ACK
                stats['fin_responded'] = True
            if stats['fin_responded'] and flags & 0x01:  # 对端发送FIN
                stats['closure_ack'] = True

            # 检测重置
            if flags & 0x04:  # RST标志
                stats['rst_received'] = True

        # 判断完整性
        stats['handshake_complete'] = all([stats['syn_sent'], stats['syn_ack_received'], stats['handshake_ack']])
        stats['closure_complete'] = all([stats['fin_initiated'], stats['fin_responded'], stats['closure_ack']])
        stats['has_payload'] = stats['bytes_transferred'] > 0
        stream_validation[stream_id] = {
            'valid_handshake': stats['handshake_complete'],
            'valid_closure': stats['closure_complete'],
            'has_payload': stats['has_payload'],
            'has_rst': stats['rst_received']
        }
    # --- 新增：将验证状态合并到原始数据 ---
    validation_df = pd.DataFrame.from_dict(stream_validation, orient='index').reset_index()
    # 4. 新增：若validation_df无数据列（仅索引），直接返回空DataFrame
    if validation_df.shape[1] != 5:  # 正常应包含"index"+4个验证字段，共5列
        print(f"⚠️ {pcap_file} 无符合条件的TCP流（握手/关闭不完整）")
        return pd.DataFrame()
    validation_df.columns = ['tcp.stream', 'valid_handshake', 'valid_closure', 'has_payload','has_rst']
    # 合并流验证状态到每个数据包
    merged_df = df.merge(validation_df, on='tcp.stream', how='left')
    filtered_df = merged_df[
        (merged_df['valid_handshake']) &
        (merged_df['valid_closure']) &
        (merged_df['has_payload']) &
        (merged_df['tcp.analysis.retransmission'].isna()) &
        (merged_df['tcp.analysis.duplicate_ack'].isna())      # 排除重复ACK
    ]
    return filtered_df
def convert_pcap_path_to_json(original_path: str) -> str:
    """
    将原始 .pcap 文件路径转换为目标 .json 文件路径，并自动创建缺失的目录。

    Args:
        original_path (str): 原始 .pcap 文件路径，例如：
            "H:/aiwendata/20241011/IPsplit/split_api_tcpdump_2024-10-11_08-09\\api_tcpdump_2024-10-11_08-09.pcap.HostPair_36-142-84-128_172-17-209-228.pcap"

    Returns:
        str: 转换后的 .json 文件路径，例如：
            "J:/Network_Traffic_Geolocation_Fingerprint/20241011/IPsplit/split_api_tcpdump_2024-10-11_08-09/api_tcpdump_2024-10-11_08-09.pcap.HostPair_36-142-84-128_172-17-209-228.json"
    """
    # 规范化路径（处理跨平台斜杠问题）
    normalized_path = os.path.normpath(original_path)

    # 替换根目录
    new_root = os.path.normpath("json_data")
    old_root = os.path.normpath("pcap_data")
    if normalized_path.startswith(old_root):
        relative_path = os.path.relpath(normalized_path, old_root)
        new_path = os.path.join(new_root, relative_path)


    # 修改文件扩展名 .pcap → .json
    base, _ = os.path.splitext(new_path)
    json_path = f"{base}.json"

    # 创建目标目录（如果不存在）
    target_dir = os.path.dirname(json_path)
    os.makedirs(target_dir, exist_ok=True)
    return json_path
def get_pcap_fingerprint(orirow,outpath, counter, lock,errorlist,error_lock):
    jsonpath = convert_pcap_path_to_json(orirow['capture_file'])
    IP=orirow['IP']
    IPint=ip_to_int(IP)
    valid_packets=tcp_flow_is_validate(pcap_file = orirow["capture_file"])
    if len(valid_packets)==0:
        print(orirow["capture_file"])
        with error_lock:
            write_error(orirow['capture_file'], errorlist)
        return
    res={}
    for index,row in valid_packets.iterrows():
        abs_time=float(row['frame.time_epoch'])
        frame_time = float(row['frame.time_relative'])  # frame.time_relative
        stream= int(row['tcp.stream'])
        window_size = int(row['tcp.window_size'])
        tcp_len = int(row['tcp.len'])
        src_ip=row['ip.src']
        ack_rtt=float(row['tcp.analysis.ack_rtt'])
        initial_rtt=float(row['tcp.analysis.initial_rtt'])

        if stream not in res:
            res[stream] = { "start_time":0,
                            "end_time":0,
                            "start_time_uplink":0,
                            "end_time_uplink":0,
                            "start_time_downlink": 0,
                            "end_time_downlink": 0,

                            "abs_time_list":[],
                            "relative_time_list":[],
                            'time_interval':[],
                            'time_interval_uplink':[],
                            'time_interval_downlink':[],

                            'burst_time_duration': [],
                            'burst_time_duration_uplink': [],
                            'burst_time_duration_downlink': [],

                            'window_size_list':[],
                            'window_size_list_uplink':[],
                            'window_size_list_downlink':[],

                            'tcp_payload_length_list':[],
                            'tcp_payload_length_list_uplink':[],
                            'tcp_payload_length_list_downlink':[],

                            'direction_list':[],

                            'burst_list':[],
                            'burst_list_uplink':[],
                            'burst_list_downlink':[],

                            "ack_rtt_list":[],
                            "ack_rtt_list_uplink": [],
                            "ack_rtt_list_downlink": [],
                            'ack_rtt_difference_list':[],
                            'ack_rtt_difference_list_uplink':[],
                            'ack_rtt_difference_list_downlink': [],
                            "initial_rtt_list":[],

                            "signal":0,
                            'uplink_signal':0,
                            'downlink_signal':0,
                            'rtt_signal':0,
                            'rtt_signal_uplink':0,
                            'rtt_signal_downlink':0,
                            'index_':0,
                            }
        if res[stream]["start_time"]==0:
            res[stream]["start_time"] = frame_time
        res[stream]["end_time"] = frame_time

        if not is_private_ip(src_ip):
            direction = 1
        else:
            direction = -1

        if res[stream]["start_time_uplink"] == 0 and direction == 1:
            res[stream]["start_time_uplink"] = frame_time
        elif direction == 1:
            res[stream]["end_time_uplink"] = frame_time
        else:
            pass

        if res[stream]["start_time_downlink"] == 0 and direction == -1:
            res[stream]["start_time_downlink"] = frame_time
        elif direction == -1:
            res[stream]["end_time_downlink"] = frame_time
        else:
            pass

        res[stream]["abs_time_list"].append(abs_time)
        res[stream]["relative_time_list"].append(frame_time)
        res[stream]["window_size_list"].append(window_size)
        res[stream]["tcp_payload_length_list"].append(tcp_len)
        res[stream]["direction_list"].append(direction)

        if not math.isnan(ack_rtt):
            res[stream]["ack_rtt_list"].append(ack_rtt)
        if not math.isnan(initial_rtt):
            res[stream]["initial_rtt_list"].append(initial_rtt)
        if res[stream]["signal"] == 0:
            res[stream]["last_time"] = frame_time
            res[stream]["burst_last_time"] = frame_time
            res[stream]["last_direction"] = direction
            res[stream]["burst"] = direction
            res[stream]["signal"]=1

            if direction == 1:
                res[stream]["burst_start_time_uplink"] = frame_time
            if direction == -1:
                res[stream]["burst_start_time_downlink"] = frame_time
        else:
            res[stream]["time_interval"].append(abs(frame_time-res[stream]["last_time"]))
            res[stream]["last_time"] = frame_time
            if res[stream]["last_direction"] == direction:
                res[stream]["burst"] = res[stream]["burst"] + direction
            else:
                res[stream]["burst_time_duration"].append(abs(frame_time-res[stream]["burst_last_time"]))
                res[stream]["burst_list"].append(res[stream]["burst"])
                if res[stream]["burst"] > 0:
                    res[stream]["burst_list_uplink"].append(res[stream]["burst"])
                    res[stream]["burst_time_duration_uplink"].append(abs(frame_time - res[stream]["burst_start_time_uplink"]))
                else:
                    res[stream]["burst_list_downlink"].append(res[stream]["burst"])
                    res[stream]["burst_time_duration_downlink"].append(abs(frame_time - res[stream]["burst_start_time_downlink"]))
                res[stream]["last_direction"] = direction
                res[stream]["burst"] = direction
                res[stream]["burst_last_time"]=frame_time

                if  direction == 1:
                    res[stream]["burst_start_time_uplink"] = frame_time

                if direction == -1:
                    res[stream]["burst_start_time_downlink"] = frame_time

        if res[stream]["uplink_signal"] == 0 and direction==1:
            res[stream]["last_time_uplink"] = frame_time
            res[stream]["uplink_signal"]=1
        elif res[stream]["uplink_signal"] == 1 and direction == 1:
            res[stream]["time_interval_uplink"].append(abs(frame_time-res[stream]["last_time_uplink"]))
            res[stream]["last_time_uplink"] = frame_time
        else:
            pass

        if res[stream]["downlink_signal"] == 0 and direction==-1:
            res[stream]["last_time_downlink"] = frame_time
            res[stream]["downlink_signal"] = 1
        elif res[stream]["downlink_signal"] == 1 and direction==-1:
            res[stream]["time_interval_downlink"].append(abs(frame_time - res[stream]["last_time_downlink"]))
            res[stream]["last_time_downlink"] = frame_time
        else:
            pass

        if res[stream]["rtt_signal"] == 0 and not math.isnan(ack_rtt):
            res[stream]["last_rtt"] = ack_rtt
            res[stream]["rtt_signal"] = 1
        elif res[stream]["rtt_signal"] == 1 and not math.isnan(ack_rtt):
            res[stream]["ack_rtt_difference_list"].append(abs(ack_rtt-res[stream]["last_rtt"]))
            res[stream]["last_rtt"] = ack_rtt
        else:
            pass

        if res[stream]["rtt_signal_uplink"] == 0 and not math.isnan(ack_rtt) and direction==1:
            res[stream]["last_rtt_uplink"] = ack_rtt
            res[stream]["rtt_signal_uplink"] = 1
        elif res[stream]["rtt_signal_uplink"] == 1 and not math.isnan(ack_rtt) and direction==1:
            res[stream]["ack_rtt_difference_list_uplink"].append(abs(ack_rtt - res[stream]["last_rtt_uplink"]))
            res[stream]["last_rtt_uplink"] = ack_rtt
        else:
            pass

        if res[stream]["rtt_signal_downlink"] == 0 and not math.isnan(ack_rtt) and direction==-1:
            res[stream]["last_rtt_downlink"] = ack_rtt
            res[stream]["rtt_signal_downlink"] = 1
        elif res[stream]["rtt_signal_downlink"] == 1 and not math.isnan(ack_rtt) and direction==-1:
            res[stream]["ack_rtt_difference_list_downlink"].append(abs(ack_rtt - res[stream]["last_rtt_downlink"]))
            res[stream]["last_rtt_downlink"] = ack_rtt
        else:
            pass

        if direction==1:
            if not math.isnan(ack_rtt):
                res[stream]["ack_rtt_list_uplink"].append(ack_rtt)
            res[stream]["tcp_payload_length_list_uplink"].append(tcp_len)
            res[stream]["window_size_list_uplink"].append(window_size)

        if direction == -1:
            if not math.isnan(ack_rtt):
                res[stream]["ack_rtt_list_downlink"].append(ack_rtt)
            res[stream]["tcp_payload_length_list_downlink"].append(tcp_len)
            res[stream]["window_size_list_downlink"].append(window_size)
    exlude=["signal",'uplink_signal','downlink_signal','rtt_signal','rtt_signal_uplink','rtt_signal_downlink',"index_","last_time","last_direction","burst","burst_start_time_uplink",
            "last_time_uplink","burst_start_time_downlink","last_time_downlink","last_rtt","last_rtt_downlink","last_rtt_uplink","burst_last_time"]
    new_res={}
    for stream in res:
        if stream not in new_res:
            new_res[stream]={}
        for key in res[stream]:
            if key not in exlude:
                new_res[stream][key] = res[stream][key]
        new_res[stream]["Payload_Throughput"] = sum(res[stream]["tcp_payload_length_list"])/ (res[stream]["end_time"]-res[stream]["start_time"])
        new_res[stream]["Packet_Throughput"] = len(res[stream]['time_interval']) / (res[stream]["end_time"] - res[stream]["start_time"])

        new_res[stream]["Payload_Throughput_uplink"] = sum(res[stream]["tcp_payload_length_list_uplink"])/ (res[stream]["end_time_uplink"]-res[stream]["start_time_uplink"])
        new_res[stream]["Packet_Throughput_uplink"] = len(res[stream]['time_interval_uplink']) / (res[stream]["end_time_uplink"] - res[stream]["start_time_uplink"])

        new_res[stream]["Payload_Throughput_downlink"] = sum(res[stream]["tcp_payload_length_list_downlink"]) / (res[stream]["end_time_downlink"] - res[stream]["start_time_downlink"])
        new_res[stream]["Packet_Throughput_downlink"] = len(res[stream]['time_interval_downlink']) / (res[stream]["end_time_downlink"] - res[stream]["start_time_downlink"])
    fingeprint = {"Avg_Duration": [], "Avg_Uplink_Duration": [], "Avg_Downlink_Duration": [],
                  "Avg_Time_Interval": [], "Avg_Time_Interval_Uplink": [], "Avg_Time_Interval_Downlink": [],
                  "Avg_Burst_Duration": [], "Avg_Burst_Duration_Uplink": [], "Avg_Burst_Duration_Downlink": [],
                  "Avg_Window_Size": [], "Avg_Window_Size_Uplink": [], "Avg_Window_Size_Downlink": [],

                  "Avg_Zero_Payload_Ratio": [],"Avg_Effective_Tcp_Payload":[],
                  "Avg_Zero_Payload_Ratio_Uplink": [], "Avg_Effective_Tcp_Payload_Uplink":[],
                  "Avg_Zero_Payload_Ratio_Downlink": [], "Avg_Effective_Tcp_Payload_Downlink":[],

                  "Up_Down_Ratio": [],  # 正方向的数量除以负方向的比值
                  "Avg_Burst_Lenth": [],  # 取正然后求均值
                  "Avg_Burst_lenth_Uplink": [],  # 取正然后求均值
                  "Avg_Burst_lenth_Downlink": [],  # 取正然后求均值

                  "Avg_Ack_Rtt": [],
                  "Avg_Ack_Rtt_Uplink": [],
                  "Avg_Ack_Rtt_Downlink": [],

                  "Avg_Initial_rtt": [],

                  "Avg_Payload_Throughput": [],
                  "Avg_Payload_Throughput_Uplink": [],
                  "Avg_Payload_Throughput_Downlink": [],

                  "Avg_Packet_Throughput": [],
                  "Avg_Packet_Throughput_Uplink": [],
                  "Avg_Packet_Throughput_Downlink": []
                  }
    for stream in new_res:
            duration= new_res[stream]["end_time"]-new_res[stream]["start_time"]
            fingeprint["Avg_Duration"].append(duration)
            duration_uplink= new_res[stream]["end_time_uplink"]-new_res[stream]["start_time_uplink"]
            fingeprint["Avg_Uplink_Duration"].append(duration_uplink)
            duration_downlink= new_res[stream]["end_time_downlink"]-new_res[stream]["start_time_downlink"]
            fingeprint["Avg_Downlink_Duration"].append(duration_downlink)

            time_interval=new_res[stream]["time_interval"]
            if len(time_interval)>0:
                fingeprint["Avg_Time_Interval"].append(iqr_based_smoothing(time_interval, window_size=3, iqr_factor=1.5))
            time_interval_uplink=new_res[stream]["time_interval_uplink"]
            if len(time_interval_uplink)>0:
                fingeprint["Avg_Time_Interval_Uplink"].append(iqr_based_smoothing(time_interval_uplink, window_size=3, iqr_factor=1.5))
            time_interval_downlink=new_res[stream]["time_interval_downlink"]
            if len(time_interval_downlink)>0:
                fingeprint["Avg_Time_Interval_Downlink"].append(iqr_based_smoothing(time_interval_downlink, window_size=3, iqr_factor=1.5))

            burst_time_duration=new_res[stream]["burst_time_duration"]
            if len(burst_time_duration)>0:
                fingeprint["Avg_Burst_Duration"].append(iqr_based_smoothing(new_res[stream]["burst_time_duration"], window_size=3, iqr_factor=1.5))
            burst_time_duration_uplink=new_res[stream]["burst_time_duration_uplink"]
            if len(burst_time_duration_uplink)>0:
                fingeprint["Avg_Burst_Duration_Uplink"].append(iqr_based_smoothing(new_res[stream]["burst_time_duration_uplink"], window_size=3, iqr_factor=1.5))
            burst_time_duration_downlink=new_res[stream]["burst_time_duration_downlink"]
            if len(burst_time_duration_downlink)>0:
                fingeprint["Avg_Burst_Duration_Downlink"].append(iqr_based_smoothing(new_res[stream]["burst_time_duration_downlink"], window_size=3, iqr_factor=1.5))

            fingeprint["Avg_Window_Size"].append(get_mean(new_res[stream]["window_size_list"]))
            fingeprint["Avg_Window_Size_Uplink"].append(get_mean(new_res[stream]["window_size_list_uplink"]))
            fingeprint["Avg_Window_Size_Downlink"].append(get_mean(new_res[stream]["window_size_list_downlink"]))

            tcp_zero_payload_ratio, tcp_effective_payload_mean=get_payload_featuree(new_res[stream]["tcp_payload_length_list"])
            if tcp_zero_payload_ratio!=-1 and tcp_effective_payload_mean !=-1:
                fingeprint["Avg_Zero_Payload_Ratio"].append(tcp_zero_payload_ratio)
                fingeprint["Avg_Effective_Tcp_Payload"].append(tcp_effective_payload_mean)

            tcp_zero_payload_ratio, tcp_effective_payload_mean=get_payload_featuree(new_res[stream]["tcp_payload_length_list_uplink"])
            if tcp_zero_payload_ratio != -1 and tcp_effective_payload_mean != -1:
                fingeprint["Avg_Zero_Payload_Ratio_Uplink"].append(tcp_zero_payload_ratio)
                fingeprint["Avg_Effective_Tcp_Payload_Uplink"].append(tcp_effective_payload_mean)

            tcp_zero_payload_ratio, tcp_effective_payload_mean=get_payload_featuree(new_res[stream]["tcp_payload_length_list_downlink"])
            if tcp_zero_payload_ratio != -1 and tcp_effective_payload_mean != -1:
                fingeprint["Avg_Zero_Payload_Ratio_Downlink"].append(tcp_zero_payload_ratio)
                fingeprint["Avg_Effective_Tcp_Payload_Downlink"].append(tcp_effective_payload_mean)

            fingeprint["Up_Down_Ratio"].append(count_positive_negative_diff(new_res[stream]["direction_list"]))

            fingeprint["Avg_Burst_Lenth"].append(get_mean_burst(new_res[stream]["burst_list"]))
            fingeprint["Avg_Burst_lenth_Uplink"].append(get_mean_burst(new_res[stream]["burst_list_uplink"]))
            fingeprint["Avg_Burst_lenth_Downlink"].append(get_mean_burst(new_res[stream]["burst_list_downlink"]))

            ack_rtt_list=new_res[stream]["ack_rtt_list"]
            if len(ack_rtt_list)>0:
                fingeprint["Avg_Ack_Rtt"].append(iqr_based_smoothing(ack_rtt_list, window_size=3, iqr_factor=1.5))
            ack_rtt_list_uplink=new_res[stream]["ack_rtt_list_uplink"]
            if len(ack_rtt_list_uplink)>0:
                fingeprint["Avg_Ack_Rtt_Uplink"].append(iqr_based_smoothing(ack_rtt_list_uplink, window_size=3, iqr_factor=1.5))
            ack_rtt_list_downlink=new_res[stream]["ack_rtt_list_downlink"]
            if len(ack_rtt_list_downlink)>0:
                fingeprint["Avg_Ack_Rtt_Downlink"].append(iqr_based_smoothing(ack_rtt_list_downlink, window_size=3, iqr_factor=1.5))

            fingeprint["Avg_Initial_rtt"].append(get_mean(new_res[stream]["initial_rtt_list"]))
            fingeprint["Avg_Payload_Throughput"].append(new_res[stream]["Payload_Throughput"])
            fingeprint["Avg_Payload_Throughput_Uplink"].append(new_res[stream]["Payload_Throughput_uplink"])
            fingeprint["Avg_Payload_Throughput_Downlink"].append(new_res[stream]["Payload_Throughput_downlink"])

            fingeprint["Avg_Packet_Throughput"].append(new_res[stream]["Packet_Throughput"])
            fingeprint["Avg_Packet_Throughput_Uplink"].append(new_res[stream]["Packet_Throughput_uplink"])
            fingeprint["Avg_Packet_Throughput_Downlink"].append(new_res[stream]["Packet_Throughput_downlink"])
    for key in fingeprint:
        if len(fingeprint[key]) == 0:
            with error_lock:
                write_error(orirow['capture_file'], errorlist)
            return
        else:
            fingeprint[key]=get_mean(fingeprint[key])
    cidr24_str, cidr24_int=ip_to_cidr24_and_int(IP)
    fingeprint["IPint"]= IPint
    fingeprint["Cidr24_Network"] = cidr24_str
    fingeprint["Cidr24_Int"] = cidr24_int
    fingeprint["jsonpath"] = jsonpath
    for key in fingeprint:
        value=fingeprint[key]
        if value!=None:
            orirow[key]=value
        else:
            with error_lock:
                write_error(orirow['capture_file'], errorlist)
            return
    with open(jsonpath, "w") as f:
        json.dump(new_res, f, indent=4)
    row_dict = orirow.to_dict()
    # 将结果保存到文件
    result_df = pd.DataFrame([row_dict])
    with lock:  # 获取锁以确保线程安全#296319
        result_df.to_csv(outpath, mode='a', header=not os.path.exists(outpath), index=False, encoding="utf-8")
        counter.value += 1
        # 打印当前 counter 的值
        print(f"当前已处理的行数: {counter.value}")
def get_feature_wrapper(args):
    # 解包参数
    row, outpath, counter, lock,errorlist,error_lock = args
    # 调用原始函数
    return get_pcap_fingerprint(row,outpath, counter, lock,errorlist,error_lock)


def get_unprocessed_data(original_df, processed_df, error_df, key_column='capture_file'):
    """
    找出未处理的行（排除已处理和出错的行）
    :param original_df: 原始完整数据
    :param processed_df: 已处理的数据
    :param error_df: 出错的数据
    :param key_column: 用于匹配的唯一键列名
    :return: 未处理的数据DataFrame
    """
    # 合并已处理和出错的文件路径
    processed_files = set()
    if len(processed_df) > 0:
        processed_files.update(processed_df[key_column])
    if len(error_df) > 0:
        processed_files.update(error_df[key_column])

    # 找出真正未处理的行
    if len(processed_files) > 0:
        unprocessed = original_df[~original_df[key_column].isin(processed_files)]
    else:
        unprocessed = original_df
    return unprocessed
def get_fingerprint():
    inputpath = "index.csv"
    outpath = "feature_data/traffic_feature.csv"
    errorpath = "feature_data/error_records.csv"
    df = pd.read_csv(inputpath)
    rowline = 0
    if not os.path.exists(errorpath):
        error_df = pd.DataFrame(columns=['capture_file'])
        error_df.to_csv(errorpath, index=False, encoding="utf-8")
        print("初始化错误记录文件")
    else:
        error_df = pd.read_csv(errorpath)
        print(f"检测到 {len(error_df)} 条错误记录")
    errorlist = error_df['capture_file'].tolist()
    if os.path.exists(outpath):
        # 如果文件存在，读取已处理的数据
        processed_df = pd.read_csv(outpath)
        rowline = len(processed_df)
        print(f"检测到已处理 {len(processed_df)} 行数据")
        # 找出未处理的行
        unprocessed_df = get_unprocessed_data(df, processed_df, error_df, key_column='capture_file')
        print(f"剩余待处理数据: {len(unprocessed_df)} 行")
    else:
        # 如果文件不存在，从头开始处理
        header_columns = ["latitude","longitude","Protocol","Address","capture_file","capture_time","IP","Country","City","District",
                          'Avg_Duration', 'Avg_Uplink_Duration', 'Avg_Downlink_Duration',
                          'Avg_Time_Interval', 'Avg_Time_Interval_Uplink', 'Avg_Time_Interval_Downlink',
                          'Avg_Burst_Duration', 'Avg_Burst_Duration_Uplink', 'Avg_Burst_Duration_Downlink',
                          'Avg_Window_Size', 'Avg_Window_Size_Uplink', 'Avg_Window_Size_Downlink',
                          'Avg_Zero_Payload_Ratio', 'Avg_Effective_Tcp_Payload', 'Avg_Zero_Payload_Ratio_Uplink',
                          'Avg_Effective_Tcp_Payload_Uplink', 'Avg_Zero_Payload_Ratio_Downlink',
                          'Avg_Effective_Tcp_Payload_Downlink',
                          'Up_Down_Ratio', 'Avg_Burst_Lenth', 'Avg_Burst_lenth_Uplink', 'Avg_Burst_lenth_Downlink',
                          'Avg_Ack_Rtt', 'Avg_Ack_Rtt_Uplink', 'Avg_Ack_Rtt_Downlink', 'Avg_Initial_rtt',
                          'Avg_Payload_Throughput', 'Avg_Payload_Throughput_Uplink', 'Avg_Payload_Throughput_Downlink',
                          'Avg_Packet_Throughput', 'Avg_Packet_Throughput_Uplink', 'Avg_Packet_Throughput_Downlink',
                          'IPint', 'Cidr24_Network', 'Cidr24_Int', 'jsonpath']
        pd.DataFrame(columns=header_columns).to_csv(outpath, index=False, encoding="utf-8")
        unprocessed_df = df
        print("初始化输出文件并写入表头，开始处理全部数据")
    with Manager() as manager:
        counter = manager.Value('i', rowline)
        lock = manager.Lock()
        error_lock = manager.Lock()
        # 准备参数，只处理未处理的数据
        tasks = [
            (row, outpath, counter, lock, errorlist, error_lock)
            for _, row in unprocessed_df.iterrows()
        ]

        with Pool(processes=45) as pool:
            results = pool.map(get_feature_wrapper, tasks)

if __name__ == "__main__":
    get_fingerprint()