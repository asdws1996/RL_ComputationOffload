import numpy as np


def evaluation(agentlist, environment, tasks, neighbors_list, change_rounds):
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
        initial_observation = [task[0], task[0], task[1], task[2], task[3], 0]
        initial_states = []
        for node in range(len(node_list)):
            if net_map[0][node] == 0:
                initial_states.append(-1)
            else:
                initial_states.append(net_states[node])
        initial_observation.extend(initial_states)
        observation = np.array(initial_observation)
        tmp_traj = []

        while True:
            if time_counter % change_rounds == 0:
                updateFlag = True
            # try process
            agentNo = int(observation[5])
            # 存储路径
            tmp_traj.append(agentNo)

            tmp_agent = agentlist[agentNo]
            actions_limit = np.array(neighbors_list[agentNo])
            action = tmp_agent.DQN.choose_action(observation, actions_limit, isEval=True)
            result = environment.perceive(observation, action, updateFlag)                              # result = [r,s']
            updateFlag = False
            cost += result[0]
            time_counter += 1
            counter = counter + 1
            observation = result[1]

            if observation[5] == node_list[-1]:
                if observation[0] == 0:
                    tmp_traj.append(observation[5])
                    valid_eps += 1
                break
            if observation[3] < 0:
                break

        trajcentory.append(tmp_traj)
        cost_his.append(cost)
        counter_his.append(counter)
#100次的均值
    cost_his = np.array(cost_his)
    res_cost = np.mean(cost_his)

    counter_his = np.array(counter_his)
    res_counter = np.mean(counter_his)
    res_valid_ratio = valid_eps / len(tasks) * 100

    return res_cost, res_counter, res_valid_ratio
