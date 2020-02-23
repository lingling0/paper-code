# 对于t时刻的网络图，形成一个Gt
# 对于每个数据d,都要建立一个这样的网络。Gd = (V, E)

# 所以对于每个数据包d, 源节点如果有多个，建一个virtualNode,连接所有的terimnal node

# 形成所有的Gd,都有不同的终端节点集合Vd

# 对于每个终端节点集合，都生成一个mst

from DRL.env import *
from model.DAG import *

dags = load_data("data/1/dag_data1.json")
t = 10
dag = dags[t]
packages = load_data("data/1/package_data1.json")
package = packages[t]
# print(dag.node_ids)
# print(dag.weight)


def get_mat(dag):
    node_num = dag.node_num
    mat = [[0] * node_num] * node_num
    nodeId_2_index = {}
    index_2_nodeId = {}
    index = 0
    for node in dag.nodes:
        nodeId_2_index[node.id] = index
        index_2_nodeId[index] = node.id
        index += 1
    for node in dag.nodes:
        for nei in node.neighborNode:
            node_id, nei_id = node.id, nei.id
            print(node_id, nei_id)
            node_index, nei_index = nodeId_2_index[node_id], nodeId_2_index[nei_id]

            mat[node_index][nei_index] = 1
            mat[nei_index][node_index] = 1
    # print(mat)
    return mat


t = 0
for dag in dags:
    m1 = get_mat(dag)
    package = packages[t]
    t += 1
    if t >= 1:
        break
    sourceNodes = package.sourceNodes
    destinationNodes = package.destinationNodes

# time, nodes, weight, node_num, node_ids = dag.time, dag.nodes, dag.weight, dag.node_num, dag.node_ids
# dag1 = DAG(time, nodes, weight, node_num, node_ids)
# mat = dag.get_mat()
# print(mat == m1)
