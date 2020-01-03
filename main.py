from env import ENV
from AgentNode import Agent
import random
from evaluation import *
from math import ceil
import matplotlib.pyplot as plt
from onehot import *
import time


MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

# task = (rest computation, total computation)


def create_taskpool(TASK_NUM):
    # 创建任务列表
    TASK_LIST = []
    for i in range(TASK_NUM):
        tmp_task = []
        tmp_task.append(np.random.randint(TASK_CPT_SCALE[0], TASK_CPT_SCALE[1]))             # task[0] : 任务需要的计算总量
        tmp_task.append(np.random.choice([i for i in range(DATA_LEN_SCALE[0], DATA_LEN_SCALE[1], 100)]))    # task[1]: 任务数据量
        # tmp_task.append((ceil(
        #                 tmp_task[0]/1                                               # 计算时延的最低要求
        #                 + (tmp_task[0] * tmp_task[1]) / (NODE_CPT_SCALE[0] * 20)    # 传输时延的最低要求
        #                 + tmp_task[0] * 50 / NODE_CPT_SCALE[0])                     # 传播时延的最低要求
        #                 ) * 2)          # task[2] : 任务时延要求（卸载时延 + 传输时延 + 传播时延）
        tmp_task.append(round(np.random.random()*0.7, 4))              # task[2] :  随机产生一个收益率λ
        des_node = np.random.randint(node_list[-10], node_list[-1]+1)
        src_node = np.random.randint(node_list[0], node_list[9]+1)
        tmp_task.append({'src_node': src_node, 'des_node': des_node})   # task[3] 生成任务的源节点和目的节点
        TASK_LIST.append(tmp_task)

    return TASK_LIST


# 随机创建模型
def create_topology(Node_Num=NODE_NUM, edge_prob=0.1):
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


def _init_observation(task, env):
    initial_observation = [task[0], task[0], task[1], task[2]]
    # 0 卸载量；1 数据量；2 收益率
    # 将任务目标节点one-hot化
    src_node = task[3]['src_node']
    present_node_OHT = one_hot_code(NODE_NUM, src_node)
    initial_observation.extend(present_node_OHT)
    des_node = task[3]['des_node']
    des_node_OHT = one_hot_code(10, des_node - 40)
    initial_observation.extend(des_node_OHT)

    tmp_dis = env.d_distance_list[des_node - 40]
    tmp_path = env.path_list[des_node - 40]
    initial_states = []
    for node in node_list:
        if env.net_map[src_node][node] == 0:
            initial_states.append(-1)
        else:
            initial_states.append(env.net_states[node])
    initial_observation.extend(initial_states)

    initial_dis = []
    for node in node_list:
        if env.net_map[src_node][node] == 0:
            initial_dis.append(-1)
        else:
            initial_dis.append(tmp_dis[node])
    initial_observation.extend(initial_dis)

    return np.array(initial_observation)







if __name__ == '__main__':
    t1 = time.time()
    train()
    t2 = time.time()
    print("Time : %5f" % (t2 - t1))
