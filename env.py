import random
import math
from onehot import *
from Dijkstra import dijkstra
import copy
from taskInfor import taskInfo
import os

# 环境设置
###################################################
change_limit = 5
delta = 1.05
NODE_NUM = 50
offload_ratio_list = [i/10 for i in range(11)]
n_features = NODE_NUM * 6 + 5
n_action = NODE_NUM * len(offload_ratio_list)
NODE_CPT_SCALE = (40, 50)      # 0.1GHz
END_NODE_CPT_SCALE = (2, 10)    # 0.1GHz
TASK_CPT_SCALE = (1, 10)    # 1e8 cycles
DISTANCE_SCALE = (50, 100)      # m
DATA_LEN_SCALE = (30, 120)    # KB
dte_params = [1,  100]
CPT_SWING_RANGE = 1
MAX_TASK_SIMULTANEOUS = 200
node_list = range(NODE_NUM)
destination = [each for each in range(40, 50)]
#################################################
#### multitask change part#######################
# class task:
#     def __init__(self):
#         # 计算总量
#         self.total_cpt = np.random.randint(TASK_CPT_SCALE[0], TASK_CPT_SCALE[1])
#         # self.cpt_eval = self.cpt_target
#         # 数据量总量
#         self.total_data = np.random.choice([i for i in range(DATA_LEN_SCALE[0], DATA_LEN_SCALE[1], 100)])
#         # 收益率λ
#         self.lamda = round(np.random.random()*0.2+0.4, 2  )
#         self.des_node = np.random.randint(node_list[-10], node_list[-1]+1)
#         self.src_node = np.random.randint(node_list[0], node_list[9]+1)
#         self.latency = np.random.choice(a=dte_params)
#### multitask change part#######################

def create_taskpool(TASK_NUM, CPTSCALE=TASK_CPT_SCALE):
    # 创建任务列表
    TASK_LIST = []
    for i in range(TASK_NUM):
        tmp_task = []
        tmp_task.append(np.random.choice([i for i in range(int(CPTSCALE[0]), int(CPTSCALE[1]+1))]))             # task[0] : 任务需要的计算总量
        tmp_task.append(np.random.choice([i for i in range(int(DATA_LEN_SCALE[0]), int(DATA_LEN_SCALE[1]+10), 10)]))    # task[1]: 任务数据量
        # tmp_task.append((ceil(
        #                 tmp_task[0]/1                                               # 计算时延的最低要求
        #                 + (tmp_task[0] * tmp_task[1]) / (NODE_CPT_SCALE[0] * 20)    # 传输时延的最低要求
        #                 + tmp_task[0] * 50 / NODE_CPT_SCALE[0])                     # 传播时延的最低要求
        #                 ) * 2)          # task[2] : 任务时延要求（卸载时延 + 传输时延 + 传播时延）
        tmp_task.append(round(np.random.random()*0.2, 4))              # task[2] :  随机产生一个收益率λ
        des_node = np.random.randint(node_list[-10], node_list[-1]+1)
        src_node = np.random.randint(node_list[0], node_list[9]+1)
        tmp_task.append({'src_node': src_node, 'des_node': des_node})   # task[3] 生成任务的源节点和目的节点

        latency_param = np.random.choice(a=dte_params)
        tmp_task.append(latency_param)                                  # task[4] latency param
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
        pre_ = [each for each in previous if each != edge_node]
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
    def __init__(self, model='Multiple'):
        # 链路信息
        self.Bandwidth = 1e7
        self.Noise = 1e-9
        self.ChannelGain = 1e-3
        self.gamma = 1e-26

        # self.a_dim = NODE_NUM + 1
        self.s_dim = n_features
        self.node_list = node_list                      # 创建节点列表
        self.net_map = create_topology()                          # 创建网络拓扑图（表示连接关系的邻接矩阵）
        self.netstate_scale = []                        # 网络节点浮动范围
        self.SWING_RANGE = CPT_SWING_RANGE                  # 网络变化幅度
        self.delta = delta                              # 正态分布相关参数
        self.destination = destination                  # 任务池中目的地集合，用于最短路算法
        self.CPT_SCALE = TASK_CPT_SCALE                 # 计算资源上下限二元组（min， max）
        # self.SCALE = CPT_SCALE
        self.task_list = create_taskpool(TASK_NUM=2000)  # 创建任务集合
        self.test_task_list = create_taskpool(TASK_NUM=500)
        self.observation = None                         # 记录当前的状态
        self.iniOffloadDistribution()                   # 记录节点卸载情况（卸载任务数量，平均卸载比例）

        # self.cpt_v = np.random.choice(range(1, 4), size=len(node_list), p=[0.2, 0.6, 0.2])      #计算速率
        self.trans_v = np.random.randint(20, 30, size=len(node_list)) * 10

        self.net_states = np.array(np.random.randint(int(END_NODE_CPT_SCALE[0]), int(END_NODE_CPT_SCALE[1]), 10).tolist() + \
                          np.random.randint(int(NODE_CPT_SCALE[0]), int(NODE_CPT_SCALE[1]), size=len(node_list)-20).tolist() + \
                        np.random.randint(int(END_NODE_CPT_SCALE[0]), int(END_NODE_CPT_SCALE[1]), 10).tolist()) / 10
        # self.net_states[0] = 0
        # self.net_states[-1] = 0

        self.distance_to_ap = np.random.randint(DISTANCE_SCALE[0], DISTANCE_SCALE[1], size=len(node_list))
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
                    distance_map[i, j] = random.randint(DISTANCE_SCALE[0], DISTANCE_SCALE[1])
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

        if model == 'Multiple':
            self.iniTimeLine(n=3)

        self.TP = 0
        self.accumlatedTS = 0

    def netUpdate(self):
        for node in range(len(self.node_list)):
            # 满足 正态分布 的网络计算力变化
            net_now = np.random.choice(self.netstate_scale[node], p=self.netstate_transfer_prob[node])
            if net_now < 0:
                net_now = 0
            elif net_now > max(self.netstate_scale[node]):
                net_now = max(self.netstate_scale[node])

            self.net_states[node] = int(net_now)

    def iniOffloadDistribution(self):
        self.offload_dis = [(0, 0) for _ in range(NODE_NUM)]

    # 计算 Transmit Energy Consumption
    def cal_TEC(self, node, L, apFlag=False):
        rate = self.trans_v[node]
        if apFlag is True:
            rate = 20
        tt = L*8 / rate     # KB -> Kb

        rau = self.Noise / np.square(self.ChannelGain) * (math.pow(2, rate/self.Bandwidth) - 1)

        return rau * tt     # J

    # 计算Offload Energy Consumption
    def cal_OEC(self, node, cycles_GHZ):
        freq = self.net_states[node]        # GHz
        cost_per_GHZ = self.gamma * np.square(freq)     # GHz ^ 2
        energy_cs = cost_per_GHZ * cycles_GHZ * 1e27    # J

        return energy_cs

    def endCostCal(self, node, task):
        # function: 计算任务结果从该节点到目标节点的代价
        # input: node, task
        # output:
        #    ec: 传输能耗
        #    delay: 传输+传送时延
        # assert tmp_task['res_cpt'] == 0, "End Check Error: task doesn't finish!"
        des_node = task[3]['des_node']
        tmp_dis = self.d_distance_list[des_node-40]
        tmp_path = self.path_list[des_node-40]
        prop_delay = tmp_dis[node] * 1e-9
        trans_delay = 0
        trans_EC = 0
        L_ = task[1]*task[2]
        while node != des_node:
            trans_delay += L_ / self.trans_v[node]
            trans_EC += self.cal_TEC(node, L_)
            node = tmp_path[node]
        # total_delay = trans_delay + prop_delay

        return trans_EC, trans_delay, prop_delay

    def perceive(self, action, task_info, updateFlag=False):
        # 获取环境反馈
        # 修改事件序列

        original_task = task_info.task
        last_tp = 0   # last_tp : 指节点最后使用的时刻
        for tmp_event in self.timeline:
            if tmp_event.last_node == None:  # last_node : 指上一跳节点
                continue
            if tmp_event.last_node == task_info.node and tmp_event.offload_TP > last_tp:
                last_tp = tmp_event.offload_TP
        wait_delay = last_tp - task_info.TP  # compute wait_time > 0
        wait_delay = wait_delay if wait_delay > 0 else 0

        # self.accumlatedTS += (task_info['TP'] - self.TP)
        # self.TP = task_info['TP']
        # print("taskTP:{}| TP: {} | TS:{}".format(task_info['TP'], self.TP, self.accumlatedTS))
        # if self.accumlatedTS > change_limit:
        #     self.netUpdate()
        #     self.accumlatedTS = 0

        res_cpt = task_info.res_cpt
        # data = self.observation[1]
        pro_ratio = original_task[2]
        total_cpt = original_task[0]
        total_amount = original_task[1]
        dte_factor = original_task[4]
        node_ = task_info.node

        next_n = np.argmax(action[0])  # action 1 from actor1 has one hot code
        # offload_delay = action[1]
        freq = self.net_states[node_]

        # offload_delay = action[1]
        offload_ratio = action[1]

        # 卸载整体cpt的比例
        offloaded_ = total_cpt * offload_ratio
        if offloaded_ > res_cpt:
            offloaded_ = res_cpt
            res_cpt = 0
        else:
            res_cpt = res_cpt - offloaded_
        # 卸载时延通过 卸载量 and 节点计算能力 决定。
        offload_delay = 0.1 * offloaded_ / freq   # 单位：秒(s)
        if updateFlag == True:  # 在确定下一跳节点周围网络状况之前，判断是否改变环境
            self.netUpdate()
        # 更新网络环境
        # next_node_net_state = []
        # next_node_dis = []

        # 计算卸载能耗跟时延
        offload_EC = self.cal_OEC(node_, offloaded_)
        L_ = total_amount * (1 - (1 - pro_ratio) * (1 - (res_cpt / total_cpt)))
        # 传输能耗

        # 传输时延
          # 单位 秒（s）
        reward_plus = 0
        if res_cpt != 0:
            done = False
            #传播时延
            prop_delay = self.distance_map[node_][next_n] * 1e-9
            trans_delay = L_ / self.trans_v[node_]
            trans_EC = self.cal_TEC(node_, L_)
        else:
            # 如果剩余CPT为0，即卸载结束，需要添加额外到目的的开销
            done = True
            reward_plus = 100
            trans_EC, trans_delay, prop_delay = self.endCostCal(node_, task_info.task)

        EC = trans_EC + offload_EC
        # 计算整体能耗
        #################################
        # if node_ == task_info['task'][3]['src_node']:
        #     EC = trans_EC + offload_EC
        # else:
        #     EC = 0
        # # 只计算终端节点的能耗

        delay = offload_delay + trans_delay + prop_delay + wait_delay
        reward = -(EC + delay * dte_factor) + reward_plus
        # reward = -(delay * dte_factor) + reward_plus
        # task_info_new = {
        #     'TP': task_info['TP'] + delay,
        #     'offload_TP': task_info['TP'] + offload_delay + wait_delay,
        #     'task': task_info['task'],
        #     'last_node': node_,
        #     'node': next_n,
        #     'res_cpt': res_cpt,
        #     'reward': task_info['reward'] + reward,
        #     'ec': task_info['ec'] + EC,
        #     'delay': task_info['delay'] + delay,
        #     'WT': task_info['WT'] + wait_delay,
        #     'TTL': task_info['TTL']-1,
        # }
        taskAction = (next_n, offload_ratio)
        task_info.updateNewTaskInfo(reward, EC, delay, offload_delay, wait_delay, taskAction)
        self.addEvent(task_info)
        self.generateObservation(task_info)
        return reward, self.observation, done, EC, delay, wait_delay

    # def reset(self,  task_info=None):
    #     # TODO(Weiz): net_states = high or low or not_near
    #     # TODO(Weiz):
    #     if task_info is None:
    #         task_info = self.newTaskInfo()
    #     self.generateObservation(task_info)
    #
    #     assert len(self.observation) == self.s_dim, 'initial observation  features does not match'
    #
    #     return self.observation, task

    def init_reward(self, task):
        factor = task[4]
        src_node = task[3]['src_node']
        dis = self.distance_to_ap[src_node]
        ec_ini = self.cal_TEC(src_node, task[1], apFlag=True)
        # delay_ini = 10 * task[1]/self.trans_v[src_node] + dis*0.1
        transfer_delay_ini = task[1] / 20
        prop_delay_ini = dis * 1e-9
        delay_ini = transfer_delay_ini + prop_delay_ini

        reward_ini = -(ec_ini + delay_ini * factor)

        # return reward_ini, ec_ini, delay_ini, transfer_delay_ini, prop_delay_ini
        return 0, 0, 0, 0, 0

    def testTaskUpdate(self):
        self.test_task_list = create_taskpool(TASK_NUM=50, CPTSCALE=self.CPT_SCALE)

    def trainTaskUpdate(self):
        self.task_list = create_taskpool(TASK_NUM=200)  # 创建任务集合

    def observation_process(self):
        tmp = copy.copy(self.observation)
        tmp[1] = (tmp[1] - DATA_LEN_SCALE[0]) / (DATA_LEN_SCALE[1] - DATA_LEN_SCALE[0])
        tmp[0] = (tmp[0] - self.CPT_SCALE[0]) / (self.CPT_SCALE[1] - self.CPT_SCALE[0])
        return tmp

    def getEvent(self):
        tmp = self.timeline.pop(0)
        # 修改observation
        self.generateObservation(tmp)
        # 返回observation
        return self.observation_process(), tmp

    def addEvent(self, task_info):
        count = 0
        for i in range(len(self.timeline)):
            count += 1
            tmp_info = self.timeline[i]
            if task_info.TP < tmp_info.TP:
                self.timeline.insert(i, task_info)
                break
        if count == len(self.timeline):
            self.timeline.insert(count, task_info)

    def iniTimeLine(self, n=5, evalFlag=False):
        self.timeline = []
        if evalFlag is False:
            tasks_indexs = np.random.choice(len(self.task_list), size=n)
            task_list = [self.task_list[i] for i in tasks_indexs]
        else:
            tasks_indexs = np.random.choice(len(self.test_task_list), size=n)
            task_list = [self.test_task_list[i] for i in tasks_indexs]
        for task in task_list:
            reward_ini, ec_ini, delay_ini, _, _ = self.init_reward(task)
            # observation, _ = self.reset(task)
            # node = task[3]['src_node']
            # task_info = {
            #     'TP': 0,
            #     'offload_TP': None,
            #     'task': task,
            #     'last_node': None,
            #     'node': node,
            #     'res_cpt': task[0],
            #     'reward': reward_ini,
            #     'ec': ec_ini,
            #     'delay': delay_ini,
            #     'WT': 0,
            #     'TTL': 10
            # }
            task_info = taskInfo(0, task, reward_ini, ec_ini, delay_ini)

            self.timeline.append(task_info)

    def newTaskInfo(self, evalFlag=False):
        # 新加入任务
        # if over is False and np.random.random() < 1 / len(self.timeline):
        if evalFlag is False:
            task = self.task_list[np.random.choice(len(self.task_list))]
        else:
            task = self.test_task_list[np.random.choice(len(self.test_task_list))]
        reward_ini, ec_ini, delay_ini, _, _ = self.init_reward(task)
        # node = task[3]['src_node']
        # task_info = {
        #     'TP': self.timeline[0]['TP'],
        #     'offload_TP': None,
        #     'task': task,
        #     'last_node': None,
        #     'node': node,
        #     'res_cpt': task[0],
        #     'reward': reward_ini,
        #     'ec': ec_ini,
        #     'delay': delay_ini,
        #     'WT': 0,
        #     'TTL': 10
        # }
        task_info = taskInfo(self.timeline[0].TP, task, reward_ini, ec_ini, delay_ini)
        return task_info

    def generateObservation(self, task_info):
        task = task_info.task

        res_cpt = task_info.res_cpt
        total_cpt = task[0]
        total_data = task[1]
        pro_ratio = task[2]
        des_node = task[3]['des_node']
        dte_param_tmp = task[4]

        dte_param = np.zeros(len(dte_params), dtype=int)
        dte_param[dte_params.index(dte_param_tmp)] = 1
        # 将任务时延敏感度化成离散值
        # 对计算资源跟数据进行scaling
        L_ = total_data * (1 - (1 - pro_ratio) * (1 - (res_cpt / total_cpt)))
        initial_observation = [res_cpt, L_, pro_ratio]
        initial_observation.extend(dte_param)
        # 0 待卸载量；1 数据量；2 收益率
        # 将d任务当前节点one-hot化
        present_node = task_info.node
        present_node_OHT = one_hot_code(NODE_NUM, present_node)
        initial_observation.extend(present_node_OHT)

        tmp_dis = self.d_distance_list[des_node - 40]
        initial_states = []
        initial_dis = []
        # for node in node_list:
        #     tmp_states = [0, 0, 0]
        #     if self.net_map[present_node][node] == 0 and node != present_node:
        #         tmp_states[0] = 1
        #         initial_dis.append(-1)
        #     else:
        #         if self.net_states[node] >= NODE_CPT_SCALE[0] / 10:
        #             tmp_states[2] = 1
        #         else:
        #             tmp_states[1] = 1
        #         initial_dis.append(tmp_dis[node])
        #     initial_states.extend(tmp_states)
        # initial_observation.extend(initial_states)
        # initial_observation.extend(initial_dis)

        for node in node_list:
            if self.net_map[present_node][node] == 0 and node != present_node:
                initial_states.append(-1)
                initial_dis.append(-1)
            else:
                initial_states.append(self.net_states[node])
                initial_dis.append(tmp_dis[node])
        initial_observation.extend(initial_states)
        initial_observation.extend(initial_dis)

        # 预计等待时长：
        # 预计等待时间 = 当前时刻 + 传输时间 + 传播时间 - 上次目标节点（i）卸载完成的时刻
        runningNode = []
        for each in self.timeline:
            if each.last_node is not None:
                runningNode.append(each.last_node)
        runningNode = set(runningNode)
        est_waiting = []
        # tmp_transfer_delay = L_ / self.trans_v[present_node]
        for i in self.node_list:
            busy_node = [0, 0, 0]
            if self.net_map[present_node][i] == 0:
                busy_node[0] = 1
                # est_waiting.append(0)
            else:
                if i not in runningNode:
                    # est_waiting.append(1)
                    busy_node[1] = 1
                else:
                    # est_waiting.append(0)
                    busy_node[2] = 1
            est_waiting.extend(busy_node)
        initial_observation.extend(est_waiting)

        assert len(initial_observation) == self.s_dim, 'initial observation  features does not match'

        self.observation = np.array(initial_observation)


if __name__ == '__main__':
    env = ENV()
    tmp_task_info = env.timeline[0]
    env.generateObservation(task_info=tmp_task_info)
    print(env.observation)