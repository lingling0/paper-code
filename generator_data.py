import random
import re
import time

import numpy as np

from model.DAG import *
from model.Node import *
from model.Package import *
from param import v2v_range, filename_data, T, data_num, data_length, bottom, grid_height, left, grid_width, width, \
    grid_num
from store import *


def generator_dag(all_nodes):
    """计算每个时刻车辆的邻居"""
    dags = []
    start = time.time()
    print("t1", time.time())
    for t in all_nodes:
        node_ids = []
        t_vehicles = all_nodes[t]
        t_vehicle_num = len(t_vehicles)
        v_ids = [n.id for n in t_vehicles]
        # print("timeslot", t, v_ids)
        t_weigth = {}
        for v in t_vehicles:
            node_ids.append(v.id)
            vid = v.id
            x, y = v.x, v.y
            neightbor = []
            v_nei = [n.id for n in v.neighborNode]
            # print(v.id)
            for tv in t_vehicles:
                tvid = tv.id
                tv_nei = [n.id for n in tv.neighborNode]
                if tvid == vid or tv_nei is None or v.id in tv_nei:
                    continue
                x1, y1 = tv.x, tv.y

                if pow((x1 - x), 2) + pow((y1 - y), 2) <= pow(v2v_range, 2):
                    if tvid not in v_nei and tvid != vid and tvid in v_ids:
                        weight = 1
                        if "i" in str(tv.id) or "i" in str(v.id):
                            weight = 2
                        if tvid not in t_weigth.keys():
                            t_weigth[tvid] = []
                        if vid not in t_weigth.keys():
                            t_weigth[vid] = []
                        neightbor.append(tv)
                        t_weigth[vid].append({tvid: weight})
                        t_weigth[tvid].append({vid: weight})
                        v.neighborNode.append(tv)
                        tv.neighborNode.append(v)

            # v.neighborNode = neightbor
            # print(v.id, [n.id for n in neightbor])
        # print(t, node_ids, t_weigth, t_vehicle_num)
        if t == 10:
            for node in all_nodes[t]:
                print(node.id, [n.id for n in node.neighborNode])
        dag = DAG(t, all_nodes[t], t_weigth, t_vehicle_num, node_ids)
        dags.append(dag)
        if t >= T:
            break
    t2 = time.time() - start
    print("t2", t2)
    store_data(all_nodes, "data/1/all_node_data1.json")
    store_data(dags, "data/1/dag_data1.json")
    print("t3", time.time() - t2)
    return all_nodes


def generator_vehicle_data():
    count = {}
    all_vehicle = {}
    t = 0
    with open('C:/Users/26271/map/vehroutes11.xml') as f:
        text = f.read()
        info = re.findall(r'<timestep time="(.*?)">(.*?)</timestep>', text, re.S)
        # print(info)
        for (timeslot, v_info) in info:
            # print(v_info)
            print("timeslot", timeslot)
            timeslot = float(timeslot)
            all_vehicle[t] = []
            v_info = v_info.strip()
            vinfo = re.findall(r'<vehicle id="(.*?)".*?speed="(.*?)".*?x="(.*?)".*?y="(.*?)"/>', v_info)
            for v1 in vinfo:
                # print(v1[2], type(v1[2]), len(v1[2]))
                id = v1[0]
                x = float(v1[2])
                speed = float(v1[1])
                y = float(v1[3])
                grid_id = get_region_id(x, y)
                if speed != 0.0:
                    if id not in count.keys():
                        count[id] = 0
                    count[id] += 1

                node = Node(id, x, y, grid_id, speed)
                all_vehicle[t].append(node)
            t += 1
            if t >= T:
                break
    print(t)
    store_data(all_vehicle, filename_data)
    return all_vehicle


def generator_infrastructure_data():
    infrastructures = []
    infrastructures.append(Node('i1', 1840, 2250, get_region_id(1840, 2250), max_capacity=100, max_through_time=10))
    infrastructures.append(Node('i2', 2550, 2240, get_region_id(2550, 2240), max_capacity=100, max_through_time=10))
    infrastructures.append(Node('i3', 1080, 2250, get_region_id(1080, 2250), max_capacity=100, max_through_time=10))
    infrastructures.append(Node('i4', 3200, 2250, get_region_id(3200, 2250), max_capacity=100, max_through_time=10))
    infrastructures.append(Node('i5', 3780, 2250, get_region_id(3780, 2250), max_capacity=100, max_through_time=10))
    infrastructures.append(Node('i6', 1035, 1500, get_region_id(1035, 1500), max_capacity=100, max_through_time=10))
    infrastructures.append(Node('i7', 1850, 1500, get_region_id(1850, 1500), max_capacity=100, max_through_time=10))
    infrastructures.append(Node('i8', 2450, 1500, get_region_id(2450, 1500), max_capacity=100, max_through_time=10))
    infrastructures.append(Node('i9', 3700, 1500, get_region_id(3700, 1500), max_capacity=100, max_through_time=10))
    infrastructures.append(Node('i10', 1820, 750, get_region_id(1820, 750), max_capacity=100, max_through_time=10))
    infrastructures.append(Node('i11', 2500, 700, get_region_id(2500, 700), max_capacity=100, max_through_time=10))
    infrastructures.append(Node('i12', 3700, 570, get_region_id(3700, 570), max_capacity=100, max_through_time=10))
    return infrastructures


def generator_node():
    infrastructures = generator_infrastructure_data()
    all_nodes = {}
    # nodes = load_data(filename_data)
    nodes = generator_vehicle_data()

    for t in nodes:
        all_nodes[t] = []
        infrastructures = generator_infrastructure_data()
        all_nodes[t].extend(infrastructures)
        all_nodes[t].extend(nodes[t])

    # store_data(all_nodes, "data/1/vehicle_data1.json")
    return all_nodes


def generator_package():
    """
    随机生成数据包
    每个timeslot产生包的数量，服从泊松分布；
    每个包产生的源节点数量，随机选择在1-4之间。
    每个包的影响范围，随机选择距离在[100-1000]之间
    每个包影响时间，随机选择在[10ms,3min]之间
    每个数据包的长度，在[10-100]之间
    每个目标车辆的截止时间，在[50ms-200ms]之间，即[5-20个tianslot]

    :return: 每个timeslot的包列表
    """
    dags = load_data("data/1/dag_data1.json")
    packages = {}
    # 产生每个timeslot生成的新增包数量，服从泊松分布
    packages_number = np.random.poisson(10, T)
    id = 0

    for t in range(T):
        print("timeslot", t)
        packages[t] = []
        packages[t] = []
        t_nodes = dags[t].nodes
        if len(t_nodes) <= 1:
            continue
        for pid in range(packages_number[t]):
            # print("pid", id)
            d_index = np.random.randint(data_num)
            d_length = data_length[d_index]
            start_time = t
            expire_time = np.random.randint(start_time, start_time + 20)
            source_node_num = np.random.randint(1, 3)
            source_nodes = random.sample(t_nodes, source_node_num)
            already_nodeIds = [node.id for node in source_nodes]
            destination_nodes = []
            destination_nodeIds = []
            for node in source_nodes:
                x, y = node.x, node.y
                inf_range = random.randint(10, 500)
                # inf_range = 500
                for v in t_nodes:
                    if v.id in already_nodeIds or v.id == node.id:
                        continue
                    # print("adsdadsa")
                    x1, y1 = v.x, v.y
                    if pow((x1 - x), 2) + pow((y1 - y), 2) <= pow(inf_range, 2):
                        deadline = np.random.randint(start_time, start_time + 15)
                        deadline = expire_time if deadline > expire_time else deadline
                        destination_node = DestinationNode(v, deadline)

                        destination_nodes.append(destination_node)
                        destination_nodeIds.append(v.id)
                        already_nodeIds.append(v.id)
            print("destination_nodeIds", destination_nodeIds)
            if len(destination_nodes) <= 0:
                continue
            package = Package(id, d_index, d_length, start_time, expire_time, source_nodes, destination_nodes,
                              destination_nodeIds)
            id += 1

            packages[t].append(package)
    store_data(packages, "data/1/package_data1.json")


def get_region_id(lat, lon):
    row_index = (lat - bottom) / grid_height
    column_index = (lon - left) / grid_width
    region_id = int(row_index) * width + int(column_index)
    if region_id > grid_num or region_id < 0:
        region_id = 0
    return region_id


# all_nodes = generator_node()
# generator_dag(all_nodes)
generator_package()
