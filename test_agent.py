from store import *
from DRL.agent import *
import tensorflow as tf
from DRL.env import *
dags = load_data("data/1/dag_data1.json")
timeslot = 99
dag = dags[timeslot]
packages = load_data("data/1/package_data1.json")
package = packages[timeslot]
total_num_destination_nodes = int(np.sum(len(p.destinationNodes) for p in package))
print(total_num_destination_nodes)

env = Environment()
env.timeslot = timeslot
env.dag = dag
env.package = package

tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.Session()
agent = ActorAgent(sess)
obs = env.observe()
agent.invoke_model(obs)
