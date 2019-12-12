from onehot import *


def evaluation(agent, environment, tasks, neighbors_list, change_rounds, segment=False):
    net_map = environment.net_map
    net_states = environment.net_states
    node_list = environment.node_list
    cost_his = []
    counter_his = []
    time_counter = 1
    trajcentory = []
    updateFlag = False
    # 记录有效的episode和无效的episode:
    valid_eps = 0

    for i in range(len(tasks)):
        task = tasks[i]
        cost = 0
        counter = 0
        # task initialization
        initial_observation = [task[0], task[0], task[1], task[2], task[3]]
        src_node = task[4]['src_node']
        present_node_OHT = one_hot_code(len(node_list), src_node)
        initial_observation.extend(present_node_OHT)
        des_node_OHT = np.zeros(10)
        des_node = task[4]['des_node']
        tmp_dis = environment.d_distance_list[des_node - 40]
        tmp_path = environment.path_list[des_node - 40]

        des_node_OHT = one_hot_code(10, des_node-40)
        initial_observation.extend(des_node_OHT)
        initial_states = []
        for node in range(len(node_list)):
            if net_map[src_node][node] == 0:
                initial_states.append(-1)
            else:
                initial_states.append(net_states[node])
        initial_observation.extend(initial_states)
        initial_dis = []
        for node in node_list:
            if net_map[src_node][node] == 0:
                initial_dis.append(-1)
            else:
                initial_dis.append(tmp_dis[node])
        initial_observation.extend(initial_dis)

        observation = np.array(initial_observation)
        tmp_traj = [[task[4]['src_node'], task[4]['des_node']]]
        # Tabu = []

        while True:
            present_node = one_hot_decode(observation[5:55])
            if segment:
                if observation[0] > 0:
                    if time_counter % change_rounds == 0:
                        updateFlag = True
                    # try process
                    # 存储路径
                    tmp_traj.append(present_node)
                    actions_limit = np.array(neighbors_list[present_node])
                    action = agent.DQN.choose_action(observation, actions_limit, isEval=False)
                else:
                    action = tmp_path[present_node]
            else:
                if time_counter % change_rounds == 0:
                    netUpdateFlag = True
                # try process
                present_node = one_hot_decode(observation[5:55])

                # 确定该节点的有效邻接节点
                actions_limit = np.array(neighbors_list[present_node])
                action = agent.DQN.choose_action(observation, actions_limit)
            result = environment.perceive(observation, action, updateFlag)  # result = [r,s']
            updateFlag = False
            cost += result[0]
            time_counter += 1
            counter = counter + 1
            observation = result[1]
            nn = one_hot_decode(observation[5:55])
            if nn == des_node:
                if observation[0] == 0:
                    tmp_traj.append(nn)
                    valid_eps += 1
                    cost_his.append(cost)
                break
            if observation[3] < 0:
                break

        trajcentory.append(tmp_traj)
        # cost_his.append(cost)
        counter_his.append(counter)

    cost_his = np.array(cost_his)
    res_cost = np.mean(cost_his-50000)

    counter_his = np.array(counter_his)
    res_counter = np.mean(counter_his)
    res_valid_ratio = valid_eps / len(tasks) * 100

    return res_cost, res_counter, res_valid_ratio
