from param import request_file_name, dag_file_name, T
from store import *
import copy

class Environment(object):
    def __init__(self):
        self.timeslot = 0
        self.packages = load_data(request_file_name)
        self.dags = load_data(dag_file_name)
        self.package = self.packages[self.timeslot]
        self.dag = self.dags[self.timeslot]
        self.finished_nodes = set()
        self.total_expired_num_nodes = 0  # 保存总的的已过期节点数量
        self.current_expired_num_nodes = 0  # 保存当前时刻的已过期节点数量
        self.total_finished_num_nodes = 0  # 保存总的目标节点数量
        self.current_finished_num_nodes = 0  # 保存当前时刻的目标节点数量
        self.total_num_destinationnodes = 0  # 保存总的的目标节点数量
        self.current_num_destinationnodes = 0 # 保存当前时刻的目标节点数量

        # 吞吐率为： self.total_finished_num_nodes/self.total_num_destinationnodes

    def observe(self):
        self.action_map1 = self.compute_act_map1()
        self.action_map2 = self.compute_act_map2()
        return self.timeslot, self.package, self.dag, self.action_map1, self.action_map2

    def reset(self):
        self.timeslot = 0
        self.package = self.packages[self.timeslot]
        self.dag = self.dags[self.timeslot]
        self.node_finished = set()

    def step(self, discard_nodes, finished_nodes, cache_nodes):
        """
        # env.step
        # 1. 从package中删除掉没有过期的包，self.unfinished_packages
        # 2. 将action2中删除package的destinationNode.
        # 3. 将cache_node加入到pakage中的sourceNode
        # 4.  如果目标节点当前已经到达节点的过期时间需要将节点从包中删除
        # 5. 将所有其他的所有节点，的waitTime+1，
        :param discard_nodes: action1,每个包，删除了哪些目标节点 {pid1:[nodes], pid2:[nodes]}
        :param finished_nodeIds: action2，每个包的目标节点，如何路由.{pid1:[成功路由的目标节点], pid2:[nodes]}
        :param cache_nodes: action3，每个包经过的节点，如何缓存,{pid1: [cache_nodes], pid2:[cache_nodes]}
        :return: 返回下一个step的环境信息；reward,该step所获得的reward；done是否结束该循环；
        """
        for p in self.package:
            finished_expired_nodeIds = []
            # 1. 从package中删除掉没有过期的包
            if p.expire_time > self.timeslot:
                self.current_expired_num_nodes += len(p.destinationNodeIds)
                finished_expired_nodeIds.extend(p.destinationNodeIds)

            # 3. 将cache_node加入到pakage中的sourceNode
            if p.id in cache_nodes.keys():
                for node in cache_nodes[p.id]:
                    p.sourceNodes.append(node)

            # 4. 如果目标节点当前已经到达节点的过期时间需要将节点从包中删除
            for d_node in p.destinationNodes:
                self.current_num_destinationnodes += 1
                if d_node.wait_time > self.timeslot:
                    p.destinationNodes.remove(d_node)
                    p.destinationNodeIds.remove(d_node.id)
                    finished_expired_nodeIds.append(d_node.id)

                # 2. 将action2中删除package的destinationNode.
                if p.id in finished_nodes.keys():
                    self.current_finished_num_nodes += len(finished_nodes[p.id])
                    for d_node1 in finished_nodes[p.id]:
                        if d_node.destination_node.id == d_node1.destination_node.id:
                            p.destinationNodes.remove(d_node)
                            p.destinationNodeIds.remove(d_node.id)
                            finished_expired_nodeIds.append(d_node.id)

            # 5. 将所有其他的非action2的节点，的waitTime+1，
            for d_node in p.destinationNodes:
                if d_node.destination_node.id not in finished_expired_nodeIds:
                    d_node.wait_time += 1
        self.total_expired_num_nodes += self.current_expired_num_nodes
        self.total_finished_num_nodes += self.current_finished_num_nodes
        self.total_num_destinationnodes += self.current_num_destinationnodes
        # 计算当前步骤的reward
        reward = self.reward_calculate()

        # 下一个step的环境
        self.timeslot += 1
        self.package += self.packages[self.timeslot]
        self.dag = self.dags[self.timeslot]

        for p in self.package:
            if len(p.destinationNodes) <= 0:
                self.package.remove(p)

        # 判断是否结束该循环
        done = True if self.timeslot > T else False

        return self.observe(), reward, done

    def reward_calculate(self):
        """
        reward怎么计算，目标是要最大化完成率。总的完成节点/总的目标节点数

        :return:
        """
        print(self.current_finished_num_nodes, self.current_finished_num_nodes)
        print(self.total_finished_num_nodes, self.total_num_destinationnodes)
        print(self.current_expired_num_nodes, self.current_num_destinationnodes)
        reward = self.current_finished_num_nodes/self.current_num_destinationnodes + \
        self.total_finished_num_nodes / self.total_num_destinationnodes - \
        self.current_expired_num_nodes/self.current_num_destinationnodes
        return reward

    def compute_act_map1(self):
        # translate action ~ [0, num_nodes_in_all_dags) to node object
        action_map = {}
        action = 0
        for p in self.package:
            for destinationNode in p.destinationNodes:
                action_map[action] = {"nodeId": destinationNode.destination_node.id, "packageId": p.id}
                action += 1
        return action_map

    def compute_act_map2(self):
        action_map = {}
        action = 0
        for node in self.dag.nodes:
            for p in self.package:
                action_map[action] = {"node": node, "package": p}
                action += 1
        return action_map
