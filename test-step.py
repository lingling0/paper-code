# test env.step

from store import *
from DRL.agent import *
import tensorflow as tf
from DRL.env import *
import copy
dags = load_data("data/1/dag_data1.json")
timeslot = 17
dag = dags[timeslot]
packages = load_data("data/1/package_data1.json")
package = packages[timeslot]
total_num_destination_nodes = int(np.sum(len(p.destinationNodes) for p in package))
print(total_num_destination_nodes)


temp_package = copy.deepcopy(package)
print("qian", len(temp_package))

for p in temp_package:
    print(p)
    for d_node in p.destinationNodes:
        print(d_node)
        print(d_node.destination_node)
    if len(p.destinationNodes) <= 1:
        # print(package)
        temp_package.remove(p)
print("hou",len(temp_package))

node = []
for p in temp_package:
    for d_node in p.destinationNodes:
        node = d_node
        break
env = Environment()
env.timeslot = timeslot
env.dag = dag
env.package = package
obs, reward, done = env.step({}, {0:[node]}, {1:[node]})
# print(obs)
print(reward)
print(done)


