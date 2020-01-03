import random
import math
from onehot import *
from Dijkstra import dijkstra

# 环境设置
###################################################

delay_reward_factor = 0.5

change_rounds = 5
delta = 1.05
NODE_NUM = 50
n_features = NODE_NUM * 3 + 14
n_action = NODE_NUM + 1
NODE_CPT_SCALE = (5, 8)
TASK_CPT_SCALE = (2, 3)
DATA_LEN_SCALE = (1000, 2000)
CPT_SWING_RANGE = 3
node_list = range(NODE_NUM)
destination = [each for each in range(40, 50)]
#################################################

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


class ENV:
    def __init__(self):
        # 链路信息
        self.Bandwidth = 1000
        self.Noise = -70
        self.ChannelGain = 1e-2
        self.gamma = 1e-19

        self.a_dim = NODE_NUM
        self.s_dim = n_features
        self.node_list = node_list                      # 创建节点列表
        self.net_map = create_topology()                          # 创建网络拓扑图（表示连接关系的邻接矩阵）
        self.netstate_scale = []                        # 网络节点浮动范围
        self.SWING_RANGE = CPT_SWING_RANGE                  # 网络变化幅度
        self.delta = delta                              # 正态分布相关参数
        self.destination = destination                  # 任务池中目的地集合，用于最短路算法
        self.CPT_SCALE = NODE_CPT_SCALE                 # 计算资源上下限二元组（min， max）
        # self.SCALE = CPT_SCALE
        self.task_list = create_taskpool(TASK_NUM=200)  # 创建任务集合
        self.test_task_list = create_taskpool(TASK_NUM=50)

        # self.cpt_v = np.random.choice(range(1, 4), size=len(node_list), p=[0.2, 0.6, 0.2])      #计算速率
        self.trans_v = np.random.randint(20, 30, size=len(node_list)) * 100

        self.net_states = np.random.randint(NODE_CPT_SCALE[0], NODE_CPT_SCALE[1], size=len(node_list))
        # self.net_states[0] = 0
        # self.net_states[-1] = 0

        # 正态分布的均方差
        for each in range(len(self.node_list)):
            tmp_cpt_scale = np.array([i for i in range(-self.SWING_RANGE, self.SWING_RANGE + 1)]) + \
                        self.net_states[each]
            self.netstate_scale.append(tmp_cpt_scale)

        # self.netstate_scale = [[net_state - 1, net_state, net_state + 1] for net_state in net_states]
        # self.netstate_transfer_prob = np.zeros([net_map.shape[0], 3])
        self.netstate_transfer_prob = []
        for each_node in range(len(self.node_list)):
            scales = self.netstate_scale[each_node]
            probes = 1 / (math.sqrt(2 * math.pi * self.delta)) * \
                np.exp(-(np.square(scales - self.net_states[each_node])) / 2 * pow(self.delta, 2))
            sum_ = np.sum(probes)
            if sum_ > 1:
                max_idx = np.argmax(probes)
                probes[max_idx] + 1 - sum_
            else:
                probes[0] = (1 - sum_) / 2 + probes[0]
                probes[-1] = (1 - sum_) / 2 + probes[-1]

            self.netstate_transfer_prob.append(probes)

        distance_map = np.copy(self.net_map)
        for i in node_list:
            for j in range(i + 1, node_list[-1] + 1):
                if distance_map[i, j] == 1:
                    distance_map[i, j] = random.randint(20, 50)
                    distance_map[j, i] = distance_map[i, j]
        self.distance_map = distance_map

        self.path_list = []
        self.d_distance_list = []
        for each in self.destination:
            tmp_dis, tmp_path = dijkstra(distance_map, each)
            self.path_list.append(tmp_path)
            self.d_distance_list.append(tmp_dis)

        self.neighbor_list = []
        # 创建邻接节点列表
        for each in node_list:
            tmp_neighbors = []
            for i in range(len(node_list)):
                if self.net_map[each][i] != 0:
                    tmp_neighbors.append(i)
            self.neighbor_list.append(tmp_neighbors)

    def netUpdate(self):
        for node in range(len(self.node_list)):
            # seed = random.random()
            # transfer_prob = self.netstate_transfer_prob[node]
            # if seed < transfer_prob[0]:
            #     self.net_states[node] = self.netstate_scale[node][0]
            # elif seed < (transfer_prob[0] + transfer_prob[1]):
            #     self.net_states[node] = self.netstate_scale[node][1]
            # else:
            #     self.net_states[node] = self.netstate_scale[node][2]

            # 满足 正态分布 的网络计算力变化
            net_now = np.random.choice(self.netstate_scale[node], p=self.netstate_transfer_prob[node])
            if net_now < 0:
                net_now = 0
            elif net_now > max(self.netstate_scale[node]):
                net_now = max(self.netstate_scale[node])

            self.net_states[node] = int(net_now)

    def cal_TEC(self, node, L):
        rate = self.trans_v[node]
        tt = L / rate
        rau = self.Noise / np.square(self.ChannelGain) * (math.pow(2, rate/self.Bandwidth) - 1)

        return rau * tt * -1e-5

    def cal_OEC(self, node, cycles_GHZ):
        freq = self.net_states[node]
        cost_per_GHZ = self.gamma * np.square(freq)
        energy_cs = cost_per_GHZ * cycles_GHZ

        return energy_cs

    def perceive(self, observation, action, EC_to_delay_factor, updateFlag=False):
        task_ = {}
        task_['res_cpt'] = observation[0]
        task_['total_cpt'] = observation[1]
        task_['data_amount'] = observation[2]
        # task_['rest_delay'] = observation[3]
        # task_['profit_ratio'] = observation[4]
        # task_['des_node'] = observation[55:65]
        task_['profit_ratio'] = observation[3]
        task_['des_node'] = observation[54:64]
        des_node = one_hot_decode(task_['des_node']) + 40
        tmp_dis = self.d_distance_list[des_node - 40]
        node_ = one_hot_decode(observation[4:54])
        next_n = action[0]  # action 1 from actor1 has one hot code
        offload_delay = action[1]

        if updateFlag == True:  # 在确定下一跳节点周围网络状况之前，判断是否改变环境
            self.netUpdate()
        # 更新网络环境
        next_node_net_state = []
        next_node_dis = []

        ##################### DQN(enhanced) ############################
        # if action > max(self.node_list):
        #     # 如果停留，则继续卸载
        #     # 更新下一跳的任务信息
        #     # 卸载量更新：
        #     if task_['res_cpt'] != 0:
        #         computation_remain = task_['res_cpt'] - self.net_states[node_]
        #         offload_ = self.net_states[node_]
        #         if computation_remain < 0:
        #             computation_remain = 0
        #             offload_ = task_['res_cpt']
        #     else:
        #         computation_remain = 0
        #         offload_ = 0
        #
        #
        #     for n in self.node_list:
        #         if self.net_map[node_][n] != 0:  # 下一跳节点就是action
        #             next_node_net_state.append(self.net_states[n])
        #             next_node_dis.append(tmp_dis[n])
        #         else:
        #             next_node_net_state.append(-1)  # next observation net_state
        #             next_node_dis.append(-1)
        #     cost_offload = offload_ * cost_per_offload  # 计算代价
        #
        #
        #     # 数据量更新：
        #     cpt_offloaded = task_['total_cpt'] - computation_remain
        #     L_ = task_['data_amount'] * (1 - (1 - task_['profit_ratio']) * (cpt_offloaded / task_['total_cpt']))
        #
        #     # 时延更新：
        #     delay_offload = 1  # 计算时延
        #     # delay_trans = L_ / self.trans_v[node_]
        #     # delay_propogation = self.distance_map[node_, action]
        #     # delay_ = task_['rest_delay'] - (delay_offload + delay_trans + delay_propogation)
        #
        #
        #     next_observation = [computation_remain, task_['total_cpt'], task_['data_amount'], task_['profit_ratio']]
        #     action_OHT = one_hot_code(len(self.node_list), node_)
        #     # # 路由代价
        #     # cost_per_bit = self.distance_map[node_][action] * 0.007
        #     # reward = - L_ * cost_per_bit
        #     # 计算卸载代价
        #     reward = - cost_offload + (delay_offload * delay_reward_factor)
        #     # if action == des_node:
        #     #     if computation_remain == 0:
        #     #         reward = reward + 50000
        #     #     else:
        #     #         reward = reward - 20000
        #
        # else:
        #     next_observation = [task_['res_cpt'], task_['total_cpt'], task_['data_amount'], task_['profit_ratio']]
        #     action_OHT = one_hot_code(len(self.node_list), action)
        #     cost_per_bit = self.distance_map[node_][action] * energy_to_distance_factor
        #     cost_trans = - task_['data_amount'] * cost_per_bit
        #     delay = task_['data_amount'] / self.trans_v[node_]
        #
        #     for n in self.node_list:
        #         if self.net_map[action][n] != 0:  # 下一跳节点就是action
        #             next_node_net_state.append(self.net_states[n])
        #             next_node_dis.append(tmp_dis[n])
        #         else:
        #             next_node_net_state.append(-1)  # next observation net_state
        #             next_node_dis.append(-1)
        #
        #     reward = cost_trans + delay * delay_reward_factor
        #     if action == des_node:
        #         if task_['res_cpt'] == 0:
        #             reward = reward + 50000
        #         else:
        #             reward = reward - 20000
        ######################### CAC ###################################
        freq = self.net_states[node_]
        if freq * offload_delay > task_['res_cpt']:
            res_cpt = 0
            offloaded_ = task_['res_cpt']
        else:
            offloaded_ = freq * offload_delay
            res_cpt = task_['res_cpt'] - offloaded_

        # 计算卸载能耗跟时延
        offload_EC = self.cal_OEC(node_, offloaded_)
        L_ = task_['data_amount'] * (1 - (1 - task_['profit_ratio']) * (1 - (res_cpt / task_['total_cpt'])))
        # 传输能耗
        trans_EC = self.cal_TEC(node_, L_)
        # 传输时延
        trans_delay = L_ / self.trans_v[node_]
        #传播时延
        prop_delay = self.distance_map[node_][next_n] * 0.1

        EC = trans_EC + offload_EC
        delay = offload_delay + trans_delay + prop_delay
        reward = -(EC + delay * EC_to_delay_factor)
        if action == des_node:
            if task_['res_cpt'] == 0:
                reward = reward + 50000
            else:
                reward = reward - 20000

        for n in self.node_list:
            if self.net_map[next_n][n] != 0:  # 下一跳节点就是action
                next_node_net_state.append(self.net_states[n])
                next_node_dis.append(tmp_dis[n])
            else:
                next_node_net_state.append(-1)  # next observation net_state
                next_node_dis.append(-1)

        next_observation = [res_cpt, task_['total_cpt'], L_, task_['profit_ratio']]
        action_OHT = one_hot_code(len(self.node_list), next_n)
        next_observation.extend(action_OHT)
        next_observation.extend(task_['des_node'])
        next_observation.extend(next_node_net_state)
        next_observation.extend(next_node_dis)
        next_observation = np.array(next_observation)
        result = (reward, next_observation)
        return result, EC, delay

