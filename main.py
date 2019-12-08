from env import ENV
from AgentNode import Agent
import random
import os
import numpy as np
from evaluation import *
from math import ceil
import matplotlib.pyplot as plt
import tensorflow as tf


# task = (rest computation, total computation)

iterations = 2000
change_rounds = 10
delta = 1
NODE_NUM = 50
n_features = 66
n_action = NODE_NUM

TASK_NUM = 100
NODE_CPT_SCALE = (20, 30)
TASK_CPT_SCALE = (50, 100)
DATA_LEN_SCALE = (1000, 2000)
CPT_SWING_RANGE = 3
node_list = range(NODE_NUM)
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


# net_map = \
# [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
# [1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
# [1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
# [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
# [0, 1, 1, 1, 0, 1, 0, 0, 1, 0],
# [0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
# [0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
# [0, 0, 0, 1, 0, 0, 0, 0, 1, 1],
# [0, 0, 0, 0, 1, 1, 0, 1, 0, 1],
# [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
# ]


# 随机创建模型
def create_topology(Node_Num=NODE_NUM, edge_prob=0.3):
    num = Node_Num
    M = np.zeros([num, num])
    previous = [0]
    for i in range(1, num):
        edge_node = random.choice(previous)
        M[i, edge_node] = 1
        M[edge_node, i] = 1
        pre_ = [each for each in previous].remove(edge_node)
        previous.append(i)
        if pre_ is None:
            continue
        else:
            for j in pre_:
                if random.random() < edge_prob:
                    M[i, j] = 1
                    M[j, i] = 1
    print(M)
    return M

print('Trying to create a connected topology')
print('-' * 50)
net_map = create_topology()

# 记录拓扑图

print("Topology Created!")
print(net_map)


# 创建环境对象
env = ENV(node_list, net_map, NODE_CPT_SCALE, CPT_SWING_RANGE, delta)

# NET_STATES = np.array([random.randint(1, 3) for _ in node_list])
# 创建邻接节点的list
neighbors_list = []
for k in node_list:
    tmp_ = []
    for n in node_list:
        if net_map[k, n] != 0:
            tmp_.append(n)
    neighbors_list.append(tmp_)

# 创建Agents
agent_list = []
for i in node_list:
    tmp_neighbors = net_map[i]
    # neighbors = NET_STATES[np.nonzero(tmp_neighbors)]
    tmp_agent = Agent(n_actions=NODE_NUM, n_features=n_features)
    agent_list.append(tmp_agent)

steps = [0 for _ in node_list]
# task

evaluation_his = []
x = []
with open('record.txt', 'w+') as fp:
    fp.write('Iteration\t\tCost\t\tCounter\t\tValidateRate\n')
    fp.write('-'*50)
    fp.write('\n')

time_counter = 1
netUpdateFlag = False
for i in range(iterations):
    task_index = np.random.randint(0, TASK_NUM)
    task = TASK_LIST[task_index]
    step_counter = 0
    initial_observation = [task[0], task[0], task[1], task[2], task[3], task[4]['src_node']]
    # task 0：卸载量；1：数据量；2：时延要求；3：λ task[4]['src_node']：表示任务的初始位置
    # 将任务目标节点one-hot化
    des_node_OHT = np.zeros(10)
    des_node = task[4]['des_node']
    des_node_OHT[des_node - 40] = 1

    initial_observation.extend(des_node_OHT)

    initial_states = []
    for node in node_list:
        if net_map[0][node] == 0:
            initial_states.append(-1)
        else:
            initial_states.append(env.net_states[node])
    initial_observation.extend(initial_states)
    observation = np.array(initial_observation)
    while True:
        if time_counter % change_rounds == 0:
            netUpdateFlag = True
        # try process
        agentNo = int(observation[5])
        tmp_agent = agent_list[agentNo]         # 选择节点/智能体

        # 确定该节点的有效邻接节点
        actions_limit = np.array(neighbors_list[agentNo])

        action = tmp_agent.DQN.choose_action(observation, actions_limit)
        result = env.perceive(observation, action, netUpdateFlag)                              # result = [r,s']
        netUpdateFlag = False
        tmp_agent.DQN.store_transition(observation, action, result[0], result[1])
        tmp_step = steps[agentNo]
        # print(result[0])

        if tmp_step > 100 and (tmp_step % 5 == 0):
            # DQN学习过程
            sample = tmp_agent.DQN.fetch_batch_sample()
            q_next = np.zeros([1, tmp_agent.DQN.n_actions])
            sample_observation_ = sample[:, -n_features:]
            for _observation in sample_observation_:
                target_node = int(_observation[5])
                q_next_value = agent_list[target_node].DQN.fetch_target(_observation)
                q_next = np.vstack((q_next, q_next_value))
            q_next = q_next[1:, :]
            tmp_agent.DQN.learn(q_next, sample, neighbors_list)

        steps[agentNo] = steps[agentNo] + 1
        step_counter += 1
        time_counter += 1
        observation = result[1]

        if observation[5] == des_node:
            break
        if observation[3] < 0:
            break

    # 保存模型
    # if i % 2000 == 0 and i > 500:
    #     for no in node_list:
    #         agent_list[no].DQN.saveModel(no, i)

    if i >= 500 and i % 100 == 0:
        res_cost, res_counter, res_valid_ration = evaluation(agent_list, env, TASK_LIST, neighbors_list, change_rounds)
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

# fig = plt.figure()
# ax1 = fig.add_subplot(311)
# ax1.plot(x, cost_his)
# ax1.set_title('cost_his')
# ax2 = fig.add_subplot(312)
# ax2.plot(x, counter_his)
# ax2.set_title('round_his')
# ax3 = fig.add_subplot(313)
# ax3.plot(x, valid_ratio_his)
# ax3.set_title('valid_episodes_ratio')
#
# plt.show()

#
# 画 DQN 的收敛情况
fig = plt.figure()
# show_list = node_list[:-1]
for i in node_list[0:4]:
    ax_tmp = fig.add_subplot(5, 2, i+1)
    agent_list[i].DQN.plot_cost(ax_tmp)
plt.show()
