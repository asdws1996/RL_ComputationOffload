import random
import math
from onehot import *

cost_per_offload = 1


class ENV:
    def __init__(self, node_list, net_map, NODE_CPT_SCALE, SWING_RANGE, delta):
        self.node_list = node_list
        self.net_map = net_map
        self.netstate_scale = []
        self.SWING_RANGE = SWING_RANGE
        self.delta = delta
        # self.SCALE = CPT_SCALE

        self.cpt_v = np.random.choice(range(1, 4), size=len(node_list), p=[0.2, 0.6, 0.2])
        self.trans_v = np.random.randint(20, 30, size=len(node_list))

        self.net_states = np.random.randint(NODE_CPT_SCALE[0], NODE_CPT_SCALE[1], size=len(node_list))
        self.net_states[0] = 0
        self.net_states[-1] = 0

        # 正态分布的均方差
        for each in range(len(self.node_list)-2):  # 除去首位 一共48条
            tmp_cpt_scale = np.array([i for i in range(-self.SWING_RANGE, self.SWING_RANGE + 1)]) + \
                        self.net_states[each+1]
            self.netstate_scale.append(tmp_cpt_scale)

        # self.netstate_scale = [[net_state - 1, net_state, net_state + 1] for net_state in net_states]
        # self.netstate_transfer_prob = np.zeros([net_map.shape[0], 3])
        self.netstate_transfer_prob = []
        for each_node in range(len(self.node_list)-2):
            scales = self.netstate_scale[each_node]
            probes = 1 / (math.sqrt(2 * math.pi * self.delta)) * \
                np.exp(-(np.square(scales - self.net_states[each_node + 1])) / 2 * pow(self.delta, 2))
            sum_ = np.sum(probes)
            probes[0] = (1 - sum_) / 2 + probes[0]
            probes[-1] = (1 - sum_) / 2 + probes[-1]
            self.netstate_transfer_prob.append(probes)

        distance_map = np.copy(net_map)
        for i in node_list:
            for j in range(i + 1, node_list[-1] + 1):
                if distance_map[i, j] == 1:
                    distance_map[i, j] = random.randint(20, 50)
                    distance_map[j, i] = distance_map[i, j]
        self.distance_map = distance_map

    def netUpdate(self):
        for node in range(len(self.node_list)-2):
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

            self.net_states[node + 1] = int(net_now)

    def perceive(self, observation, action, updateFlag=False):
        task_ = {}
        task_['res_cpt'] = observation[0]
        task_['total_cpt'] = observation[1]
        task_['data_amount'] = observation[2]
        task_['rest_delay'] = observation[3]
        task_['profit_ratio'] = observation[4]
        task_['des_node'] = observation[55:65]
        des_node = one_hot_decode(task_['des_node']) + 40

        node_ = one_hot_decode(observation[5:55])
        # net_state = observation[4:]

        # 更新下一跳的任务信息
        # 卸载量更新：
        if task_['res_cpt'] != 0:
            computation_remain = task_['res_cpt'] - self.net_states[node_]
            offload_ = self.net_states[node_]
            if computation_remain < 0:
                computation_remain = 0
                offload_ = task_['res_cpt']
        else:
            computation_remain = 0
            offload_ = 0
        cost_offload = offload_ * cost_per_offload  # 计算代价
        delay_offload = offload_ / self.cpt_v[node_]  # 计算时延

        # 数据量更新：
        cpt_offloaded = task_['total_cpt'] - computation_remain
        L_ = task_['data_amount'] * (1 - (1 - task_['profit_ratio']) * (cpt_offloaded / task_['total_cpt']))

        # 时延更新：
        delay_trans = L_ / self.trans_v[node_]
        delay_propogation = self.distance_map[node_, action]
        delay_ = task_['rest_delay'] - (delay_offload + delay_trans + delay_propogation)

        if updateFlag == True:  # 在确定下一跳节点周围网络状况之前，判断是否改变环境
            self.netUpdate()
        # 更新网络环境
        next_node_net_state = []
        for n in self.node_list:
            if self.net_map[action][n] != 0:  # 下一跳节点就是action
                next_node_net_state.append(self.net_states[n])
            else:
                next_node_net_state.append(-1)  # next observation net_state
        next_observation = [computation_remain, task_['total_cpt'], L_, delay_, task_['profit_ratio']]
        action_OHT = one_hot_code(len(self.node_list), action)
        next_observation.extend(action_OHT)
        next_observation.extend(task_['des_node'])
        next_observation.extend(next_node_net_state)
        next_observation = np.array(next_observation)
        # 路由代价
        cost_per_bit = self.distance_map[node_][action] * 0.007
        reward = - L_ * cost_per_bit
        # 计算卸载代价
        reward = reward - cost_offload

        if action == des_node:
            if computation_remain == 0:
                reward = reward + 50000
            else:
                reward = reward - 20000

        if delay_ < 0:  # 任务超时
            reward = reward - 300

        result = (reward, next_observation)

        return result
