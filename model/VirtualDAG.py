from model.Node import VirtualNode
# from DRL.env import *

def get_mat(nodes):
    node_num = len(nodes)
    mat = [[0] * node_num] * node_num
    nodeId_2_index = {}
    index_2_nodeId = {}
    index = 0
    for node in nodes:
        nodeId_2_index[node.id] = index
        index_2_nodeId[index] = node.id
        index += 1
    for node in nodes:
        for nei in node.neighborNode:
            node_id, nei_id = node.id, nei.id
            node_index, nei_index = nodeId_2_index[node_id], nodeId_2_index[nei_id]

            mat[node_index][nei_index] = 1
            mat[nei_index][node_index] = 1
    return mat

def get_graph_for_package(dag, package):

    virtualNode = VirtualNode()
    virtualNode.neighborNode = package.sourceNodes
    nodes = dag.nodes
    nodes = nodes + virtualNode
    print(len(nodes))





def test():
    dags = load_data("data/1/dag_data1.json")
    t = 10
    dag = dags[t]
    packages = load_data("data/1/package_data1.json")
    package = packages[t]
    get_graph_for_package(dag, package)
test()