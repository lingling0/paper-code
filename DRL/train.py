import multiprocessing as mp
import time

from DRL.agent import *
from DRL.env import *
from DRL.params import *
from util import *


def invoke_model(actor_agent, obs, exp):
    """
    调用强化学习模型，返回action=<a1, a2, a3>
    :param actor_agent:
    :param obs:
    :param exp:
    :return:
    """
    # parse observation
    timeslot, package, dag, action_map1, action_map2 = obs

    # invoking the learning model
    discard_act_probs, cache_act_probs, discard_acts, cache_acts, \
    destination_node_inputs, cache_inputs, dag, package, node_valid_mask1, node_valid_mask2 = \
        actor_agent.invoke_model(obs)

    if sum(node_valid_mask1[0, :]) == 0:
        # no node is valid to assign
        return None

    # node_act should be valid
    assert node_valid_mask2[0, discard_acts[0]] == 1

    # parse node action，得到对应的package和node。
    discard_node = action_map1[discard_acts[0]]

    # node_act should be valid
    assert node_valid_mask2[0, cache_acts[0]] == 1

    # parse node action
    cache_node = action_map2[discard_acts[0]]

    # TODO: SMT problem求解，返回finishedNode.即成功路由的节点集合。
    finished_nodes = {}

    # store experience
    exp['destination_node_inputs'].append(destination_node_inputs)
    exp['cache_inputs'].append(cache_inputs)
    exp['dag'].append(dag)
    exp['package'].append(package)
    exp['node_valid_mask1'].append(node_valid_mask1)
    exp['node_valid_mask2'].append(node_valid_mask2)

    return discard_node, finished_nodes, cache_node


def train_agent(agent_id, param_queue, reward_queue, adv_queue, gradient_queue):
    # set up environment
    env = Environment()

    # gpu configuration
    config = tf.ConfigProto(
        device_count={'GPU': args.worker_num_gpu},
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=args.worker_gpu_fraction))

    sess = tf.compat.v1.Session(config=config)

    # set up actor agent
    actor_agent = ActorAgent(sess, args.node_input_dim)

    # collect experiences
    while True:
        # get parameters from master
        (actor_params, seed, entropy_weight) = \
            param_queue.get()

        # synchronize model
        actor_agent.set_params(actor_params)

        # reset environment
        env.reset()

        # set up storage for experience
        exp = {'destination_node_inputs': [], 'cache_inputs': [], \
               'dag': [], 'gcn_masks': [], \
               'summ_mats': [], 'package': [], \
               'node_valid_mask1': [], \
               'node_valid_mask2': [], "reward": []}

        try:
            # run experiment
            obs = env.observe()
            done = False


            while not done:
                # 调用强化学习模型，返回所选择的action.<action_1, action_2,action_3>
                discard_nodes, finished_nodes, cache_nodes = invoke_model(actor_agent, obs, exp)

                obs, reward, done = env.step( discard_nodes, finished_nodes, cache_nodes)

                if finished_nodes is not None:
                    # valid action, store reward and time
                    exp['reward'].append(reward)
                    exp['wall_time'].append(env.wall_time.curr_time)
                elif len(exp['reward']) > 0:
                    # Note: if we skip the reward when node is None
                    # (i.e., no available actions), the sneaky
                    # agent will learn to exhaustively pick all
                    # nodes in one scheduling round, in order to
                    # avoid the negative reward
                    exp['reward'][-1] += reward

            # report reward signals to master
            assert len(exp['node_inputs']) == len(exp['reward'])
            reward_queue.put(
                [exp['reward'],
                 len(env.finished_job_dags),
                 np.mean([j.completion_time - j.start_time \
                          for j in env.finished_job_dags]),
                 env.wall_time.curr_time >= env.max_time])

            # get advantage term from master
            batch_adv = adv_queue.get()

            if batch_adv is None:
                # some other agents panic for the try and the
                # main thread throw out the rollout, reset and
                # try again now
                continue

            # compute gradients
            actor_gradient, loss = compute_actor_gradients(
                actor_agent, exp, batch_adv, entropy_weight)

            # report gradient to master
            gradient_queue.put([actor_gradient, loss])

        except AssertionError:
            # ask the main to abort this rollout and
            # try again
            reward_queue.put(None)
            # need to still get from adv_queue to
            # prevent blocking
            adv_queue.get()


def main():
    # create result and model folder
    # create_folder_if_not_exists(args.result_folder)
    # create_folder_if_not_exists(args.model_folder)

    # initialize communication queues
    params_queues = [mp.Queue(1) for _ in range(args.num_agents)]
    reward_queues = [mp.Queue(1) for _ in range(args.num_agents)]
    adv_queues = [mp.Queue(1) for _ in range(args.num_agents)]
    gradient_queues = [mp.Queue(1) for _ in range(args.num_agents)]

    # set up training agents
    agents = []
    for i in range(args.num_agents):
        agents.append(mp.Process(target=train_agent, args=(
            i, params_queues[i], reward_queues[i],
            adv_queues[i], gradient_queues[i])))

    # start training agents
    for i in range(args.num_agents):
        agents[i].start()

    # gpu configuration
    config = tf.ConfigProto(
        device_count={'GPU': args.master_num_gpu},
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=args.master_gpu_fraction))

    sess = tf.compat.v1.Session(config=config)

    # set up actor agent
    actor_agent = ActorAgent(sess, args.node_input_dim)

    # tensorboard logging
    # tf_logger = TFLogger(sess, [
    #     'actor_loss', 'entropy', 'value_loss', 'episode_length',
    #     'average_reward_per_second', 'sum_reward', 'reset_probability',
    #     'num_jobs', 'reset_hit', 'average_job_duration',
    #     'entropy_weight'])

    # store average reward for computing differential rewards
    avg_reward_calculator = AveragePerStepReward(
        args.average_reward_storage_size)

    # initialize entropy parameters
    entropy_weight = args.entropy_weight_init

    # initialize episode reset probability
    reset_prob = args.reset_prob

    # ---- start training process ----
    for ep in range(1, args.num_ep):
        print('training epoch', ep)

        # synchronize the model parameters for each training agent
        actor_params = actor_agent.get_params()

        # send out parameters to training agents
        for i in range(args.num_agents):
            params_queues[i].put([actor_params, args.seed + ep, entropy_weight])

        # storage for advantage computation
        all_rewards, all_diff_times, all_times, \
        all_num_finished_jobs, all_avg_job_duration, \
        all_reset_hit, = [], [], [], [], [], []

        t1 = time.time()

        # get reward from agents
        any_agent_panic = False

        for i in range(args.num_agents):
            result = reward_queues[i].get()

            if result is None:
                any_agent_panic = True
                continue
            else:
                batch_reward, batch_time, \
                num_finished_jobs, avg_job_duration, \
                reset_hit = result

            diff_time = np.array(batch_time[1:]) - \
                        np.array(batch_time[:-1])

            all_rewards.append(batch_reward)
            all_diff_times.append(diff_time)
            all_times.append(batch_time[1:])
            all_num_finished_jobs.append(num_finished_jobs)
            all_avg_job_duration.append(avg_job_duration)
            all_reset_hit.append(reset_hit)

            avg_reward_calculator.add_list_filter_zero(
                batch_reward, diff_time)

        t2 = time.time()
        print('got reward from workers', t2 - t1, 'seconds')

        if any_agent_panic:
            # The try condition breaks in some agent (should
            # happen rarely), throw out this rollout and try
            # again for next iteration (TODO: log this event)
            for i in range(args.num_agents):
                adv_queues[i].put(None)
            continue

        # compute differential reward
        all_cum_reward = []
        avg_per_step_reward = avg_reward_calculator.get_avg_per_step_reward()
        for i in range(args.num_agents):
            if args.diff_reward_enabled:
                # differential reward mode on
                rewards = np.array([r - avg_per_step_reward * t for \
                                    (r, t) in zip(all_rewards[i], all_diff_times[i])])
            else:
                # regular reward
                rewards = np.array([r for \
                                    (r, t) in zip(all_rewards[i], all_diff_times[i])])

            cum_reward = discount(rewards, args.gamma)

            all_cum_reward.append(cum_reward)

        t3 = time.time()
        print('advantage ready', t3 - t2, 'seconds')

        actor_gradients = []
        all_action_loss = []  # for tensorboard
        all_entropy = []  # for tensorboard
        all_value_loss = []  # for tensorboard

        for i in range(args.num_agents):
            (actor_gradient, loss) = gradient_queues[i].get()

            actor_gradients.append(actor_gradient)
            all_action_loss.append(loss[0])
            all_entropy.append(-loss[1] / \
                               float(all_cum_reward[i].shape[0]))
            all_value_loss.append(loss[2])

        t4 = time.time()
        print('worker send back gradients', t4 - t3, 'seconds')

        actor_agent.apply_gradients(
            aggregate_gradients(actor_gradients), args.lr)

        t5 = time.time()
        print('apply gradient', t5 - t4, 'seconds')


        # decrease entropy weight
        entropy_weight = decrease_var(entropy_weight,
                                      args.entropy_weight_min, args.entropy_weight_decay)

        # decrease reset probability
        reset_prob = decrease_var(reset_prob,
                                  args.reset_prob_min, args.reset_prob_decay)

        if ep % args.model_save_interval == 0:
            actor_agent.save_model(args.model_folder + \
                                   'model_ep_' + str(ep))

    sess.close()


if __name__ == '__main__':
    main()
