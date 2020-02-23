# from store import *
#
# dags = load_data("data/1/dag_data1.json")
# dag = dags[0]
# packages = load_data("data/1/package_data1.json")
# package = packages[0]
# print(dag)
# config = tf.ConfigProto(
#     device_count={'GPU': args.worker_num_gpu},
#     gpu_options=tf.GPUOptions(
#         per_process_gpu_memory_fraction=args.worker_gpu_fraction))
#
# sess = tf.compat.v1.Session(config=config)
# agent = ActorAgent(sess)
# obs = 0, package, dag
# agent.translate_state(obs)
import numpy as np
node_inputs = np.zeros([5, 5 + 1])
print(node_inputs)
b = [2, 4]
node_inputs[1, 1:3] = b
print(node_inputs)
print(node_inputs[1,3])
a = [1, 2, 3]
b = [4, 5]
a.extend()