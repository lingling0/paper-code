import numpy as np
import tensorflow as tf

from DRL.params import *


class ActorAgent(object):
    def __init__(self, sess, node_dim=5, package_dim=3, location_embedding_size=5, eps=1e-6,
                 optimizer=tf.compat.v1.train.AdamOptimizer, scope='actor_agent'):
        self.sess = sess
        self.package_dim = package_dim
        self.node_dim = node_dim
        self.location_embedding_size = location_embedding_size
        self.eps = eps
        self.optimizer = optimizer
        self.scope = scope

        # network1 inputs, 特征为[waitTime, 节点位置，package_feature]
        self.destination_node_inputs = tf.compat.v1.placeholder(tf.float32, [None, self.node_dim])

        # network2 inputs, 特征为[waitTime, 节点位置，package_feature, 以及缓存次数]
        self.cache_inputs = tf.compat.v1.placeholder(tf.float32, [None, self.node_dim + 1])

        # valid mask for node action ([batch_size, total_num_nodes])，保存当前时刻可以选择的所有destination-node
        # 计算为：对于所有的包，所有的目标节点[node_index, node],node_valid_mask[index]=1,否则为0
        self.node_valid_mask1 = tf.compat.v1.placeholder(tf.float32, [None, None])

        # t时刻的所有节点*所有数据，用于第二层网络输入
        self.node_valid_mask2 = tf.compat.v1.placeholder(tf.float32, [None, None])

        self.discard_act_probs, self.cache_act_probs = self.actor_network(self.destination_node_inputs,
                                                                          self.cache_inputs,
                                                                          self.node_valid_mask1, self.node_valid_mask2)
        logits = tf.math.log(self.discard_act_probs)
        noise = tf.random.uniform(tf.shape(logits))
        # node_acts [batch_size, 1]，选择多少个？tf.math.argmax只能选择最大的一个，可能不太合理。之后怎么修改？！请多想并试试
        # tf.math.argmax( , axis=1)表示以行为单位返回，最大值所在的索引
        self.discard_acts = tf.math.argmax(logits - tf.math.log(-tf.math.log(noise)), 1)

        logits = tf.math.log(self.cache_act_probs)
        noise = tf.random.uniform(tf.shape(logits))
        # node_acts [batch_size, 1]，选择多少个？缓存多少个？替换节点的缓存，选择概率高的。只能缓存上限
        self.cache_acts = tf.math.argmax(logits - tf.math.log(-tf.math.log(noise)), 1)

        # Selected action for node, 0-1 vector ([batch_size, total_num_nodes])
        self.discard_act_vec = tf.compat.v1.placeholder(tf.float32, [None, None])
        # Selected action for job, 0-1 vector ([batch_size, num_jobs, num_limits])
        self.cache_act_vec = tf.compat.v1.placeholder(tf.float32, [None, None, None])

        # advantage term (from Monte Calro or critic) ([batch_size, 1])
        self.adv = tf.compat.v1.placeholder(tf.float32, [None, 1])

        # use entropy to promote exploration, this term decays over time
        self.entropy_weight = tf.compat.v1.placeholder(tf.float32, ())

        # select node action probability
        self.selected_node_prob = tf.reduce_sum(tf.multiply(
            self.discard_act_probs, self.discard_act_vec),
            axis=1, keepdims=True)

        # select job action probability
        self.selected_cache_prob = tf.reduce_sum(tf.reduce_sum(tf.multiply(
            self.cache_act_probs, self.cache_act_vec),
            axis=1), keepdims=True)

        # actor loss due to advantge (negated)
        self.adv_loss = tf.reduce_sum(tf.multiply(
            tf.math.log(self.selected_node_prob * self.selected_cache_prob + \
                        self.eps), -self.adv))

        # node_entropy
        self.node_entropy = tf.reduce_sum(tf.multiply(
            self.discard_act_probs, tf.math.log(self.discard_act_probs + self.eps)))

        # job entropy
        self.job_entropy = \
            tf.reduce_sum(tf.multiply(
                self.selected_cache_prob, tf.math.log(self.selected_cache_prob + self.eps)))
        # entropy loss
        self.entropy_loss = self.node_entropy + self.job_entropy

        # normalize entropy
        self.entropy_loss /= \
            (tf.math.log(tf.cast(tf.shape(self.discard_act_probs)[1], tf.float32)) + \
             tf.math.log(tf.cast(tf.shape(self.cache_act_probs)[1], tf.float32)))
        # normalize over batch size (note: adv_loss is sum)
        # * tf.cast(tf.shape(self.discard_act_probs)[0], tf.float32)

        # define combined loss
        self.act_loss = self.adv_loss + self.entropy_weight * self.entropy_loss

        # get training parameters
        self.params = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

        # operations for setting network parameters
        self.input_params, self.set_params_op = \
            self.define_params_op()

        # actor gradients
        self.act_gradients = tf.gradients(self.act_loss, self.params)

        # adaptive learning rate
        self.lr_rate = tf.compat.v1.placeholder(tf.float32, shape=[])

        # actor optimizer
        self.act_opt = self.optimizer(self.lr_rate).minimize(self.act_loss)

        # apply gradient directly to update parameters
        self.apply_grads = self.optimizer(self.lr_rate). \
            apply_gradients(zip(self.act_gradients, self.params))

        # network paramter saver
        self.saver = tf.compat.v1.train.Saver(max_to_keep=args.num_saved_models)
        self.sess.run(tf.compat.v1.global_variables_initializer())

        if args.saved_model is not None:
            self.saver.restore(self.sess, args.saved_model)

    def actor_network(self, destination_node_inputs, cache_inputs, node_valid_mask1, node_valid_mask2):
        batch_size1 = tf.shape(node_valid_mask1)[0]
        # (1) reshape node inputs to batch format,
        location_embedding = tf.compat.v1.get_variable('location_embedding', [100, self.location_embedding_size])
        location1 = tf.nn.embedding_lookup(location_embedding, tf.cast(destination_node_inputs[:, 0], tf.int32))
        location2 = tf.nn.embedding_lookup(location_embedding, tf.cast(cache_inputs[:, 0], tf.int32))
        destination_node_inputs = tf.concat([location1, destination_node_inputs[:, 1:]], axis=1)
        cache_inputs = tf.concat([location2, cache_inputs[:, 1:]], axis=1)
        destination_node_inputs_reshape = destination_node_inputs
        # destination_node_inputs_reshape = tf.reshape(destination_node_inputs, [batch_size1, self.node_dim])
        batch_size2 = tf.shape(node_valid_mask2)[0]
        # cache_inputs_reshape = tf.reshape(cache_inputs, [batch_size2, self.node_dim])
        cache_inputs_reshape = cache_inputs
        # (2) actor neural network
        print("batch", batch_size1, batch_size2)

        with tf.compat.v1.variable_scope(self.scope):
            # input_1 = tf.keras.Input(shape=destination_node_inputs_reshape)
            node_hid_0 = tf.keras.layers.Dense(32, activation='relu')(destination_node_inputs_reshape)
            node_hid_1 = tf.keras.layers.Dense(16, activation='relu')(node_hid_0)
            node_hid_2 = tf.keras.layers.Dense(8, activation='relu')(node_hid_1)
            outputs_1 = tf.keras.layers.Dense(1, activation=None)(node_hid_2)

            # reshape the output dimension (batch_size, total_num_nodes)
            discard_outputs = tf.reshape(outputs_1, [batch_size1, -1])
            discard_outputs = tf.nn.softmax(discard_outputs, axis=-1)
            # 经过一番解析。计算action2

            # 训练2
            # input_2 = tf.keras.layers.Input(shape=cache_inputs_reshape)
            node_hid_0 = tf.keras.layers.Dense(32, activation='relu')(cache_inputs_reshape)
            node_hid_1 = tf.keras.layers.Dense(16, activation='relu')(node_hid_0)
            node_hid_2 = tf.keras.layers.Dense(8, activation='relu')(node_hid_1)
            outputs_2 = tf.keras.layers.Dense(1, activation=None)(node_hid_2)

            cache_outputs = tf.reshape(outputs_2, [batch_size2, -1])
            cache_outputs = tf.nn.softmax(cache_outputs, axis=-1)

        return discard_outputs, cache_outputs

    def predict(self, destination_node_inputs, cache_inputs,
                node_valid_mask1, node_valid_mask2):
        return self.sess.run([self.discard_act_probs, self.cache_act_probs,
                              self.discard_acts, self.cache_acts], feed_dict=
                            {i: d for i, d in zip([self.destination_node_inputs] + [self.cache_inputs] + [self.node_valid_mask1] + \
                              [self.node_valid_mask2], [destination_node_inputs] + [cache_inputs] + [node_valid_mask1] + [node_valid_mask2])
                            })

    def invoke_model(self, obs):
        # implement this module here for training
        # (to pick up state and action to record)
        destination_node_inputs, cache_inputs, dag, package, node_valid_mask1, node_valid_mask2 = self.translate_state(
            obs)
        # invoke learning model
        if len(destination_node_inputs) <= 0:
            print("no destination nodes need to request data")
            return None
        discard_act_probs, cache_act_probs, discard_acts, cache_acts = \
            self.predict(destination_node_inputs, cache_inputs, node_valid_mask1, node_valid_mask2)
        print(discard_act_probs, cache_act_probs)
        print(discard_acts, cache_acts)
        return discard_act_probs, cache_act_probs, discard_acts, cache_acts, destination_node_inputs, cache_inputs, dag, package, node_valid_mask1, node_valid_mask2, self.discard_act_vec, self.cache_act_vec

    def translate_state(self, obs):
        """
        将obs装成网络的输入destination_node_inputs, cache_inputs,
        # network1 inputs, 特征为[waitTime, 节点位置，package_feature]
        # network2 inputs, 特征为[waitTime, 节点位置，package_feature, 以及缓存次数]
        :return:
        """
        timeslot, package, dag, action_map1, action_map2 = obs
        total_num_destination_nodes = int(np.sum(len(p.destinationNodes) for p in package))
        total_num_nodes = dag.node_num
        total_num_packages = sum(1 for _ in package)
        destination_node_inputs = np.zeros([total_num_destination_nodes, self.node_dim])
        node_inputs = np.zeros([total_num_nodes * total_num_packages, self.node_dim + 1])
        package_features = np.zeros([total_num_packages, self.package_dim])

        package_index = 0
        node_index = 0
        requested_node = {}
        node_valid_mask1 = np.zeros([1, total_num_destination_nodes])
        node_valid_mask2 = np.zeros([1, total_num_nodes * total_num_packages])
        ids = []
        for p in package:
            package_features[package_index, 0] = p.length
            package_features[package_index, 1] = p.expire_time
            package_features[package_index, 2] = p.start_time

            for d in p.destinationNodes:
                ids.append(d.destination_node.id)
                requested_node[d.destination_node.id] = node_index
                destination_node_inputs[node_index, 0] = d.wait_time
                destination_node_inputs[node_index, 1] = d.destination_node.region_id
                destination_node_inputs[node_index, 2:] = package_features[package_index, :]
                node = d.destination_node
                if node.through_times >= node.max_through_times:
                    node_valid_mask1[node_index] = 0
                node_index += 1

            package_index += 1

        for node in dag.nodes:
            for p in package:
                if node.id in p.destinationNodeIds:
                    index = requested_node[node.id]
                    node_inputs[index, :5] = destination_node_inputs[index, :]
                    # node缓存数据d的次数
                    if p.data not in node.cache_time.keys():
                        node_inputs[index, 5] = 0
                    else:
                        node_inputs[index, 5] = node.cache_time[p.data]
                else:
                    node_inputs[node_index, 0] = 0
                    node_inputs[node_index, 1] = node.region_id
                    if p.data not in node.cache_time.keys():
                        node_inputs[node_index, 5] = 0
                    else:
                        node_inputs[node_index, 5] = node.cache_time[p.data]
                    node_index += 1

                if node.through_times >= node.max_through_times:
                    node_valid_mask2[node_index] = 0
        return destination_node_inputs, node_inputs, dag, package, node_valid_mask1, node_valid_mask2

    def define_params_op(self):
        # define operations for setting network parameters
        input_params = []
        for param in self.params:
            input_params.append(
                tf.compat.v1.placeholder(tf.float32, shape=param.get_shape()))
        set_params_op = []
        for idx, param in enumerate(input_params):
            set_params_op.append(self.params[idx].assign(param))
        return input_params, set_params_op

    def get_params(self):
        return self.sess.run(self.params)

    def set_params(self, input_params):
        self.sess.run(self.set_params_op, feed_dict={
            i: d for i, d in zip(self.input_params, input_params)
        })

    def save_model(self, file_path):
        self.saver.save(self.sess, file_path)
