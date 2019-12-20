from onehot import *
from CentralTrainDistributedExecution import _init_observation


def evaluation(agent, environment, tasks, neighbors_list, change_rounds, segment=False):
    net_map = environment.net_map
    net_states = environment.net_states
    node_list = environment.node_list
    cost_his = []
    counter_his = []
    latency_his = []
    time_counter = 1
    trajcentory = []
    updateFlag = False
    # 记录有效的episode和无效的episode:
    # valid_eps = 0

    for i in range(len(tasks)):
        task = tasks[i]
        cost = 0
        counter = 0
        delay = 0
        # task initialization
        observation = _init_observation(task, environment)
        des_node = task[3]['des_node']
        tmp_path = environment.path_list[des_node - 40]
        tmp_traj = [[task[3]['src_node'], task[3]['des_node']]]
        # Tabu = []

        while True:
            present_node = one_hot_decode(observation[4:54])
            actions_limit = neighbors_list[present_node]
            actions_limit = [each for each in neighbors_list[present_node]]
            actions_limit.append(len(node_list))
            actions_limit = np.array(actions_limit)
            if segment:
                if observation[0] > 0:
                    if time_counter % change_rounds == 0:
                        updateFlag = True
                    # try process
                    # 存储路径
                    tmp_traj.append(present_node)

                    action = agent.DQN.choose_action(observation, actions_limit, isEval=False)
                else:
                    action = tmp_path[present_node]
            else:
                if time_counter % change_rounds == 0:
                    netUpdateFlag = True
                # try process
                present_node = one_hot_decode(observation[4:54])

                # 确定该节点的有效邻接节点

                action = agent.DQN.choose_action(observation, actions_limit)

            if action > max(environment.node_list):
                delay_ = 1
            else:
                delay_ = observation[2] / environment.trans_v[present_node]
                counter = counter + 1
            result = environment.perceive(observation, action, updateFlag)  # result = [r,s']

            updateFlag = False
            cost += result[0]
            delay += delay_
            time_counter += 1

            observation = result[1]
            nn = one_hot_decode(observation[4:54])
            if nn == des_node:
                if observation[0] == 0:
                    tmp_traj.append(nn)
                    cost_his.append(cost)
                    latency_his.append(delay)
                break

        trajcentory.append(tmp_traj)
        # cost_his.append(cost)
        counter_his.append(counter)

    cost_his = np.array(cost_his)
    res_cost = np.mean(cost_his-50000)
    latency_his = np.array(latency_his)
    res_latency = np.mean(latency_his)

    counter_his = np.array(counter_his)
    res_counter = np.mean(counter_his)
    # res_valid_ratio = valid_eps / len(tasks) * 100

    return res_cost, res_counter, res_latency
