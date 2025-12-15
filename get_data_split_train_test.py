import os
import pandas as pd
import ipaddress
import json

import numpy as np
import pandas as pd

from json.decoder import JSONDecodeError
def get_cross_district(path):
    '''
    具体步骤：

    读取数据，选择指定列，并过滤flow_count>=1的记录。
    将数据与自己进行合并（merge），以IP为键，排除相同File Path（即同一个文件内的记录不构成跨城市）的记录。
    合并后，我们会得到同一个IP在两个不同城市（city）的记录，一条来自左边的记录（我们称为记录A），一条来自右边的记录（记录B）。我们计算两个记录的时间差（这里用日期），排除右边日期小于左边日期的情况（即只保留右边日期>=左边日期的记录）。
    对于同一个IP和同一个左边的File Path（即同一个起始记录），可能有多个右边记录（同一个IP在不同城市的其他记录），我们只保留右边记录中日期与左边记录最接近（即时间差最小）的一条。

    :param path:
    :return:
    '''
    df=pd.read_csv(path)

    column_mapping = {
        "latitude": "lat",
        "longitude": "lon",
        "District": "district",
        "City": "city"
    }

    # 4. 执行修改
    df.rename(columns=column_mapping, inplace=True)

    print(len(df))
    # 转换日期为datetime格式
    df["capture_time"] = pd.to_datetime(
        df["capture_time"],
        format="%Y-%m-%d_%H-%M-%S",  # 匹配 "2024-05-23_19-57-33" 格式
    )
    def ip_to_cidr28(ip):
        try:
            ip_obj = ipaddress.ip_address(ip)
            if ip_obj.version == 4:
                # 对于IPv4，转换为CIDR/28格式
                network = ipaddress.ip_network(f"{ip}/24", strict=False)
                return str(network)
            elif ip_obj.version == 6:
                # 对于IPv6，转换为CIDR/56格式（类似IPv4的/28）
                network = ipaddress.ip_network(f"{ip}/56", strict=False)
                return str(network)
        except:
            return None

    # 应用IP转换函数
    df['cidr24'] = df['IP'].apply(ip_to_cidr28)
    # 2. 自连接查找跨城市IP
    # 复制两份数据，分别作为左表和右表
    left = df.copy().add_suffix('_left')
    right = df.copy().add_suffix('_right')

    # 基于IP进行自连接
    merged = pd.merge(left, right,
                      left_on=['cidr24_left','city_left'],
                      right_on=['cidr24_right',"city_right"],
                      suffixes=('', '_y'))
    print(list(merged.columns))
    # 3. 筛选有效跨城市记录
    # 排除相同文件路径的记录
    merged = merged[merged["capture_time_left"] != merged["capture_time_right"]]
    merged['Date_diff'] = (merged['capture_time_right'] - merged['capture_time_left']).dt.total_seconds()
    merged = merged[merged['Date_diff'] > 0]  # 排除负日期差

    merged["cross_district"] = 0
    # 筛选不同城市的记录
    cross_district_mask = merged['district_left'] != merged['district_right']
    merged.loc[cross_district_mask, 'cross_district'] = 1  # 跨城标记为1
    cross_district=merged[merged["cross_district"]==1]

    unique_ips = list(set(cross_district['cidr24_left']).union(set(cross_district['cidr24_right'])))
    # 2. 获取所有城市（包括left和right）并去重

    merged=merged[merged["cidr24_left"].isin(unique_ips)]
    print(len(merged),len(merged[merged["cross_district"]==1]),len(merged[merged["cross_district"]==0]))
    return merged

def get_flow_count():
    # 1. 配置文件路径（原文件路径+新文件保存路径）
    data_path = "G:\\Network_Traffic_Geolocation\\street_geolocation\\pcap_reprocess\\traffic_finger_city.csv"
    save_path = "G:\\Network_Traffic_Geolocation\\street_geolocation\\pcap_reprocess\\traffic_finger_city_with_flow.csv"

    # 2. 读取原始CSV文件
    print("正在读取原始文件...")

    df = pd.read_csv(data_path)




    # 4. 定义函数：解析jsonpath字符串，统计键值对数量（处理异常格式）
    def count_json_key_value(json_file_path):
        with open(json_file_path, "r", encoding="utf-8") as f:  # 指定utf-8编码避免乱码
            json_data = json.load(f)


            return len(json_data)


    # 5. 应用函数生成flow_count列（新增列）
    print("正在解析jsonpath列，生成flow_count...")
    df["flow_count"] = df["jsonpath"].apply(count_json_key_value)

    # 6. 查看处理结果（前5行示例）
    print("\n处理结果示例（前5行）：")
    print(df[["jsonpath", "flow_count"]].head())

    # 7. 保存新文件
    print(f"\n正在保存新文件到：{save_path}")
    try:
        df.to_csv(save_path, index=False, encoding="utf-8")
        print(f"新文件保存成功！新增'flow_count'列后，数据形状：{df.shape}（行×列）")
    except Exception as e:
        print(f"保存文件时发生错误：{str(e)}")
        exit()
keylist=['time_interval', 'time_interval_uplink',
                   'time_interval_downlink', 'burst_time_duration', 'burst_time_duration_uplink',
                   'burst_time_duration_downlink', 'window_size_list', 'window_size_list_uplink',
                   'window_size_list_downlink', 'tcp_payload_length_list', 'tcp_payload_length_list_uplink',
                   'tcp_payload_length_list_downlink', 'direction_list', 'burst_list', 'burst_list_uplink',
                   'burst_list_downlink', 'ack_rtt_list', 'ack_rtt_list_uplink', 'ack_rtt_list_downlink',
                   'ack_rtt_difference_list', 'ack_rtt_difference_list_uplink', 'ack_rtt_difference_list_downlink',
                   'initial_rtt_list', 'Payload_Throughput', 'Packet_Throughput', 'Payload_Throughput_uplink',
                   'Packet_Throughput_uplink', 'Payload_Throughput_downlink', 'Packet_Throughput_downlink']
def get_finger(featurelist,directionlist):
    finger=[]
    if len(featurelist) < len(directionlist):
        directionlist = directionlist[(len(directionlist)-len(featurelist)):]
    elif len(featurelist) == len(directionlist):
        directionlist = directionlist
    else:
        directionlist = directionlist+ [0] * (len(featurelist) - len(directionlist))
    for i in range(len(featurelist)):
        finger.append(featurelist[i]*directionlist[i])
    finger=pad_or_truncate(finger, target_length=23, pad_value=0)
    return finger
def get_burst_finger(featurelist,directionlist):
    finger=[]
    if len(featurelist) < len(directionlist):
        directionlist = directionlist[(len(directionlist)-len(featurelist)):]
    elif len(featurelist) == len(directionlist):
        directionlist = directionlist
    else:
        directionlist = directionlist+ [0] * (len(featurelist) - len(directionlist))
    for i in range(len(featurelist)):
        if directionlist[i]>0:
            finger.append(featurelist[i]*1)
        elif directionlist[i]<0:
            finger.append(featurelist[i] * (-1))
        else:
            finger.append(featurelist[i]*0)
    finger=pad_or_truncate(finger, target_length=13, pad_value=0)
    return finger
def get_ack_rtt(ack_rtt,ack_rtt_up,ack_rtt_down):
    finger=[]
    for i in ack_rtt:
        if i in ack_rtt_up:
            finger.append(i*1)
        elif i in ack_rtt_down:
            finger.append(i*(-1))
        else:
            finger.append(i*0)
    finger=pad_or_truncate(finger, target_length=13, pad_value=0)
    return finger
def pad_or_truncate(sequence: list, target_length: int, pad_value=0) -> list:
    """序列填充或截断到固定长度"""
    if len(sequence) >= target_length:
        return sequence[:target_length]  # 截断
    return sequence + [pad_value] * (target_length - len(sequence))  # 填充
def process_finger(path):
    all_fingerprints = []
    fs_finger_all = []
    df_finger_all = []
    with open(path) as f:
        jsonp = json.load(f)
        max_streams = 3
        processed_streams = 0
        for key in jsonp.keys():
            fingerprint = []
            if processed_streams > max_streams:
                break
            data = jsonp[key]
            time_interval = data['time_interval']
            window_size_list = data['window_size_list']
            tcp_payload_length_list = data['tcp_payload_length_list']
            direction_list = data['direction_list']
            burst_time_duration = data['burst_time_duration']
            burst_list = data['burst_list']
            ack_rtt_list = data['ack_rtt_list']
            ack_rtt_list_uplink = data['ack_rtt_list_uplink']
            ack_rtt_list_downlink = data['ack_rtt_list_downlink']
            initial_rtt_list = data['initial_rtt_list']
            Payload_Throughput = data['Payload_Throughput']
            Packet_Throughput = data['Packet_Throughput']
            Payload_Throughput_uplink = data['Payload_Throughput_uplink']
            Packet_Throughput_uplink = data['Packet_Throughput_uplink']
            Payload_Throughput_downlink = data['Payload_Throughput_downlink']
            Packet_Throughput_downlink = data['Packet_Throughput_downlink']
            fs_finger=pad_or_truncate(tcp_payload_length_list, target_length=20, pad_value=0)
            df_finger=pad_or_truncate(direction_list, target_length=20, pad_value=0)
            fingerprint = fingerprint + get_finger(time_interval, direction_list)
            fingerprint = fingerprint + get_finger(window_size_list, direction_list)
            fingerprint = fingerprint + get_finger(tcp_payload_length_list, direction_list)
            fingerprint = fingerprint + pad_or_truncate(burst_list, target_length=13, pad_value=0)
            fingerprint = fingerprint + get_burst_finger(burst_time_duration, burst_list)
            fingerprint = fingerprint + get_ack_rtt(ack_rtt_list, ack_rtt_list_uplink, ack_rtt_list_downlink)
            fingerprint = fingerprint + [float(np.mean(initial_rtt_list)), Payload_Throughput, Payload_Throughput_uplink,
                                         Payload_Throughput_downlink, Packet_Throughput, Packet_Throughput_uplink,
                                         Packet_Throughput_downlink]
            all_fingerprints.append(fingerprint)
            fs_finger_all.append(fs_finger)
            df_finger_all.append(df_finger)
            processed_streams += 1

    feature_matrix = np.array(all_fingerprints)
    mean_vals = np.mean(feature_matrix, axis=0)

    fs_feature_matrix = np.array(fs_finger_all)
    fs_mean_vals = np.mean(fs_feature_matrix, axis=0)

    df_feature_matrix = np.array(df_finger_all)
    df_mean_vals = np.mean(df_feature_matrix, axis=0)
    # 组合成512维特征向量 [min1, max1, mean1, median1, min2, max2, ...]
    aggregated = np.empty(115)
    for i in range(115):
        aggregated[i] = mean_vals[i]
    aggregated=aggregated.tolist()

    aggregated_fs = np.empty(20)
    for i in range(20):
        aggregated_fs[i] = fs_mean_vals[i]
    aggregated_fs=aggregated_fs.tolist()

    aggregated_df = np.empty(20)
    for i in range(20):
        aggregated_df[i] = df_mean_vals[i]
    aggregated_df=aggregated_df.tolist()
    return aggregated,aggregated_fs,aggregated_df
def get_data(path):
    df=pd.read_csv(path)
    # 计算两列的中位数
    df["finger"]=None
    df["fs_finger"]=None
    df["df_finger"]=None
    # 使用列表暂存结果
    fingers = []
    fs_fingers = []
    df_fingers = []
    for _, row in df.iterrows():
        finger,fs_finger,df_finger=process_finger(row["jsonpath"])
        fingers.append(finger)
        fs_fingers.append(fs_finger)
        df_fingers.append(df_finger)
        # 批量赋值
    df["finger"] = fingers
    df["fs_finger"] = fs_fingers
    df["df_finger"] = df_fingers
    return df

def get_district_train_test(path):
    import pandas as pd
    import numpy as np

    # 读取数据并筛选有效样本
    df = pd.read_csv(path)#"../data/cross_district_aliyun_finger.csv"
    df=df[df["city_left"].isin(["北京市"])]
    df.rename(columns={'lon_left': 'lng_left','lon_right': 'lng_right'}, inplace=True)
    df = df[
        (df["IP_left"] == df["IP_right"]) &
        (df["district_left"].notna()) &
        (df["district_right"].notna())
    ].copy()

    # 确保存在cross_district列（假设0表示同区，1表示跨区）
    if "cross_district" not in df.columns:
        # 若不存在，自动生成（根据district_left和district_right是否相同）
        df["cross_district"] = (df["district_left"] != df["district_right"]).astype(int)
        print("自动生成cross_district标签（0=同区，1=跨区）")

    # 查看原始标签分布
    label_counts = df["cross_district"].value_counts()
    print(f"原始数据标签分布：\n{label_counts}\n")

    # 按cross_district分层，每层内按20%抽样（保证分层平衡）
    test_frac = 0.2  # 测试集比例
    test_indices = []

    # 对每个标签（0和1）分别处理
    for label in [0, 1]:
        # 筛选当前标签的子集
        label_subset = df[df["cross_district"] == label].copy()
        if len(label_subset) == 0:
            print(f"警告：标签{label}无样本，跳过该层抽样")
            continue

        # 按原逻辑分组（IP_left和capture_time_left），确保每组内样本不重复抽样
        groups = label_subset.groupby(['IP_left', 'capture_time_left'])
        # 对每个分组，按Date_diff排序后取索引
        group_indices = [
            group.sort_values(by='Date_diff').index.tolist()
            for _, group in groups
        ]

        # 对分组列表进行抽样（抽取20%的组）
        np.random.seed(42)  # 固定随机种子，保证结果可复现
        n_groups = len(group_indices)
        n_test_groups = int(n_groups * test_frac)  # 测试集应包含的组数
        # 随机选择测试集的组索引
        test_group_idx = np.random.choice(n_groups, size=n_test_groups, replace=False)
        # 收集选中组的所有样本索引
        for idx in test_group_idx:
            test_indices.extend(group_indices[idx])

    # 去重（避免同一行被多个组选中，理论上不会发生）
    test_indices = list(set(test_indices))
    # 提取测试集
    test_df = df.loc[test_indices].reset_index(drop=True)
    # 训练集为剩余样本
    train_df = df[~df.index.isin(test_indices)].reset_index(drop=True)

    # 验证测试集比例和标签平衡
    total_samples = len(df)
    test_ratio = len(test_df) / total_samples
    test_label_counts = test_df["cross_district"].value_counts()
    print(f"测试集总样本数：{len(test_df)}，占比：{test_ratio:.2%}（目标20%）")
    print(f"测试集标签分布：\n{test_label_counts}")
    print(f"训练集总样本数：{len(train_df)}")



    return test_df, train_df

def main():
    path="feature_data/traffic_feature.csv"
    finger_df = get_data(path)
    finger_df.to_csv("feature_data/traffic_finger.csv", index=False)
    cross_df=get_cross_district('feature_data/traffic_finger.csv')
    cross_df.to_csv("feature_data/cross_district_finger.csv", index=False)
    test_df, train_df=get_district_train_test("feature_data/cross_district_finger.csv")
    test_df.to_csv('feature_data/cross_district_test.csv', index=False, encoding='utf-8-sig')
    train_df.to_csv('feature_data/cross_district_train.csv', index=False, encoding='utf-8-sig')
    print("测试集和训练集已保存")
main()