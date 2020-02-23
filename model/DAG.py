class DAG(object):
    def __init__(self, time, nodes, weight, node_num, node_ids):
        self.time = time
        self.nodes = nodes
        self.node_num = node_num
        self.weight = weight
        self.node_ids = node_ids

    def get_mat(self, nodes):
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

    # def get_graph_for_package(self, package):
    #     virtualNode = Node()
    #     nodes = self.nodes
    #     mat = self.get_mat()

