from env import ENV
from AgentNode import Agent
import random
from evaluation import *
from math import ceil
import matplotlib.pyplot as plt
from onehot import *
import time

iterations = 100000
change_rounds = 5
delta = 1.05
NODE_NUM = 50
n_features = NODE_NUM * 3 + 15
n_action = NODE_NUM
NODE_CPT_SCALE = (20, 30)
TASK_CPT_SCALE = (100, 150)
DATA_LEN_SCALE = (1000, 2000)
CPT_SWING_RANGE = 10
node_list = range(NODE_NUM)
# task = (rest computation, total computation)



def create_taskpool(TASK_NUM):
    # 创建任务列表
    TASK_LIST = []
    for i in range(TASK_NUM):
        tmp_task = []
        tmp_task.append(np.random.randint(TASK_CPT_SCALE[0], TASK_CPT_SCALE[1]))             # task[0] : 任务需要的计算总量
        tmp_task.append(np.random.choice([i for i in range(DATA_LEN_SCALE[0], DATA_LEN_SCALE[1], 100)]))    # task[1]: 任务数据量
        tmp_task.append((ceil(
                        tmp_task[0]/1                                               # 计算时延的最低要求
                        + (tmp_task[0] * tmp_task[1]) / (NODE_CPT_SCALE[0] * 20)    # 传输时延的最低要求
                        + tmp_task[0] * 50 / NODE_CPT_SCALE[0])                     # 传播时延的最低要求
                        ) * 2)          # task[2] : 任务时延要求（卸载时延 + 传输时延 + 传播时延）
        tmp_task.append(round(np.random.random()*0.7, 4))              # task[3] :  随机产生一个收益率λ
        des_node = np.random.randint(node_list[-10], node_list[-1]+1)
        src_node = np.random.randint(node_list[0], node_list[9]+1)
        tmp_task.append({'src_node': src_node, 'des_node': des_node})   # task[4] 生成任务的源节点和目的节点
        TASK_LIST.append(tmp_task)

    return TASK_LIST


# 随机创建模型
def create_topology(Node_Num=NODE_NUM, edge_prob=0.15):
    print('Trying to create a connected topology')
    print('-' * 50)
    num = Node_Num
    M = np.zeros([num, num])
    previous = [0]
    for i in range(1, num):
        edge_node = random.choice(previous)
        M[i, edge_node] = 1
        M[edge_node, i] = 1
        pre_ = [each for each in previous]
        pre_.remove(edge_node)
        previous.append(i)
        if pre_ is None:
            continue
        else:
            for j in pre_:
                if random.random() < edge_prob:
                    M[i, j] = 1
                    M[j, i] = 1
    print(M)
    # 记录拓扑图
    print("Topology Created!")

    return M


def _init_env():
    net_map = create_topology()
    # 创建环境对象
    destination = [i for i in range(40, 50)]
    env = ENV(node_list, net_map, NODE_CPT_SCALE, CPT_SWING_RANGE, delta, destination)
    # 创建任务池
    task_list = create_taskpool(TASK_NUM=200)
    test_task_list = create_taskpool(TASK_NUM=50)

    return net_map, env, task_list, test_task_list


def train(segment=False):
    # 环境初始化
    net_map, env, task_list, test_task_list = _init_env()

    # NET_STATES = np.array([random.randint(1, 3) for _ in node_list])
    # 创建邻接节点的list
    neighbors_list = []
    for k in node_list:
        tmp_ = []
        for n in node_list:
            if net_map[k, n] != 0:
                tmp_.append(n)
        neighbors_list.append(tmp_)

    # 创建Agent
    agent = Agent(n_actions=NODE_NUM, n_features=n_features)
    step = 0
    # task

    # 记录评估
    evaluation_his = []
    x = []
    with open('record.txt', 'w+') as fp:
        fp.write('Iteration\t\tCost\t\tCounter\t\tValidateRate\n')
        fp.write('-'*50)
        fp.write('\n')

    time_counter = 1
    netUpdateFlag = False
    for i in range(iterations):
        task_index = np.random.randint(0, len(task_list))
        task = task_list[task_index]
        step_counter = 0
        initial_observation = [task[0], task[0], task[1], task[2], task[3]]
        # task 0：卸载量；1：数据量；2：时延要求；3：λ task[4]['src_node']：表示任务的初始位置
        # 将任务目标节点one-hot化
        src_node = task[4]['src_node']
        present_node_OHT = one_hot_code(NODE_NUM, src_node)
        initial_observation.extend(present_node_OHT)
        des_node = task[4]['des_node']
        des_node_OHT = one_hot_code(10, des_node-40)
        initial_observation.extend(des_node_OHT)
        tmp_dis = env.d_distance_list[des_node-40]
        tmp_path = env.path_list[des_node-40]

        initial_states = []
        for node in node_list:
            if net_map[src_node][node] == 0:
                initial_states.append(-1)
            else:
                initial_states.append(env.net_states[node])
        initial_observation.extend(initial_states)

        initial_dis = []
        for node in node_list:
            if net_map[src_node][node] == 0:
                initial_dis.append(-1)
            else:
                initial_dis.append(tmp_dis[node])
        initial_observation.extend(initial_dis)

        observation = np.array(initial_observation)
        # Tabu = []
        while True:
            present_node = one_hot_decode(observation[5:5 + NODE_NUM])
            if segment:
                if observation[0] > 0:
                    if time_counter % change_rounds == 0:
                        netUpdateFlag = True
                    # try process

                    # 确定该节点的有效邻接节点
                    actions_limit = np.array(neighbors_list[present_node])
                    # # 在邻接节点中去除已经走过的节点
                    # Tabu.append(present_node)
                    # tmp_actions_limit = []
                    # for act in actions_limit:
                    #     if act not in Tabu:
                    #         tmp_actions_limit.append(act)
                    #
                    # # 如果无路可走就结束回合
                    # if tmp_actions_limit:
                    #     actions_limit = np.array(tmp_actions_limit)
                    # else:
                    #     break

                    action = agent.DQN.choose_action(observation, actions_limit)
                else:
                    action = tmp_path[present_node]

            else:
                if time_counter % change_rounds == 0:
                    netUpdateFlag = True
                # try process
                present_node = one_hot_decode(observation[5:5 + NODE_NUM])

                # 确定该节点的有效邻接节点
                actions_limit = np.array(neighbors_list[present_node])
                action = agent.DQN.choose_action(observation, actions_limit)
            result = env.perceive(observation, action, netUpdateFlag)  # result = [r,s']
            netUpdateFlag = False
            agent.DQN.store_transition(observation, action, result[0], result[1])
            # print(result[0])
            step += 1
            step_counter += 1
            time_counter += 1
            observation = result[1]

            if one_hot_decode(observation[5:55]) == des_node:
                break
            if observation[3] < 0:
                break

        # 保存模型
        # if i % 2000 == 0 and i > 500:
        #     for no in node_list:
        #         agent_list[no].DQN.saveModel(no, i)
        if i > 50 and (i % 5 == 0):
            # DQN学习过程
            agent.DQN.learn(neighbors_list)

        if i >= 500 and i % 100 == 0:
            res_cost, res_counter, res_valid_ration = evaluation(agent, env, test_task_list, neighbors_list, change_rounds,
                                                                 segment=False)
            with open('record.txt', 'a+') as fp:
                fp.write('%d\t\t%f\t\t%f\t\t%f\n' % (i, res_cost, res_counter, res_valid_ration))
            evaluation_his.append([res_cost, res_counter, res_valid_ration])
            x.append(i)

        # if i > 500 and (i % 1000 == 0):
        #     print(agent_list[0].DQN.fetch_eval(np.array(initial_observation)))

        print("the %d time cost %d rounds!" % (i+1, step_counter+1))

    # 记录
    cost_his = [each[0] for each in evaluation_his]
    counter_his = [each[1] for each in evaluation_his]
    valid_ratio_his = [each[2] for each in evaluation_his]

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.plot(x, cost_his)
    ax1.set_title('cost_his')
    ax2 = fig.add_subplot(222)
    ax2.plot(x, counter_his)
    ax2.set_title('round_his')
    ax3 = fig.add_subplot(223)
    ax3.plot(x, valid_ratio_his)
    ax3.set_title('valid_episodes_ratio')
    ax4 = fig.add_subplot(224)
    agent.DQN.plot_cost(ax4)
    plt.show()

    #
    # 画 DQN 的收敛情况
    # fig = plt.figure()
    # # show_list = node_list[:-1]
    # agent.DQN.plot_cost()
    # plt.show()


if __name__ == '__main__':
    t1 = time.time()
    train()
    t2 = time.time()
    print("Time : %5f" % (t2 - t1))
