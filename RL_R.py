import numpy as np
import pandas as pd
import tensorflow as tf
import os
from onehot import *
np.random.seed(1)
tf.set_random_seed(1)


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.8,
            e_greedy=0.6,
            replace_target_iter=50,
            memory_size=500,
            batch_size=64,
            e_greedy_increment=0.005,
            output_graph=False,
            prioritized=False
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.prioritized = prioritized
        # replace counter:
        self.replace_counter = 0
        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        if self.prioritized:
            self.memory = Memory(capacity=memory_size)
        else:
            self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        # tf.get_collection(key, scope=None)
        # 用来获取一个名称是‘key’的集合中的所有元素，返回的是一个列表
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []
        self.saver = tf.train.Saver(max_to_keep=1)

    def _build_net(self):
        tf.reset_default_graph()
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        if self.prioritized:
            self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 70, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2  # [batch_size,self.n_action]

        with tf.variable_scope('loss'):
            if self.prioritized:
                self.abs_errors = tf.reduce_sum(tf.abs(self.q_target - self.q_eval), axis=1)  # for updating Sumtree
                self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.q_eval))
            else:
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self, s, a, r, s_):
        if self.prioritized:  # prioritized replay
            transition = np.hstack((s, [a, r], s_))
            self.memory.store(transition)  # have high priority for newly arrived transition

        else:  # random replay
            if not hasattr(self, 'memory_counter'):
                self.memory_counter = 0

            transition = np.hstack((s, [a, r], s_))

            # replace the old memory with new memory
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition

            self.memory_counter += 1

    def choose_action(self, observation, actions_limit, isEval=False):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]  # shape=(1,n_features)

        if isEval:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            actions_validate = np.array([actions_value[0][index] for index in actions_limit])
            max_value = np.max(actions_validate)  # 未加axis＝,返回一个索引数值
            action = np.argwhere(actions_value[0] == max_value)[0]

        else:
            if np.random.uniform() < self.epsilon:
                # forward feed the observation and get q value for every actions
                actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
                actions_validate = np.array([actions_value[0][index] for index in actions_limit])
                max_value = np.max(actions_validate)  # 未加axis＝,返回一个索引数值
                action = np.argwhere(actions_value[0] == max_value)[0]
            else:
                action = np.random.choice(actions_limit, size=1)
                # action = np.random.randint(0, self.n_actions)
        return int(action)

    def learn(self, neighbor_list):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')
        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
            # sample batch memory from all memory
        else:
            if self.memory_counter > self.memory_size:
                sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            else:
                sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
            batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                # [s, a, r, s_]
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })
        # double DQN
        #
        # q_double_net = self.sess.run(self.q_eval, feed_dict={self.s: batch_memory[:, -self.n_features:]})
        # q_next_max = np.zeros(1)
        # for i in range(len(batch_memory)):
        #     observation_ = batch_memory[i]
        #     tmp_nn = one_hot_decode(observation_[5:55])
        #     tmp_des = one_hot_decode(observation_[55:65]) + 40
        #     if tmp_nn == tmp_des:
        #         tmp_next_max = 0
        #     else:
        #         limit = neighbor_list[tmp_nn]
        #         q_double_net_limit = [q_double_net[i][index] for index in limit]
        #         max_index = np.argmax(q_double_net_limit)
        #         tmp_next_max = q_next[i, max_index]
        #     q_next_max = np.hstack((q_next_max, tmp_next_max))
        # q_next_max = q_next_max[1:]

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)  # action   astype(int) 转换数组的数据类型
        reward = batch_memory[:, self.n_features + 1]  # reward

        # 在得到的q_next 中选择limit范围内value最大的
        if q_next.shape[0] == batch_memory.shape[0]:
            q_next_max = np.zeros(1)  # 初始化
            for i in range(q_next.shape[0]):
                observation_ = batch_memory[i, -self.n_features:]  # 取出下一个状态
                node_num = one_hot_decode(observation_[5:55])
                observation_DN = int(np.argwhere(observation_[55:65] == 1))  # 判断下个状态是否抵达终点
                if node_num == observation_DN:
                    tmp_max = 0
                else:
                    actions_limits = neighbor_list[node_num]
                    tmp_max = np.max(np.array([q_next[i, index] for index in actions_limits]))
                q_next_max = np.hstack((q_next_max, tmp_max))
            q_next_max = q_next_max[1:]
        else:
            print("train error! the sample dimension doesn't match the values")
            return -1

        q_target[batch_index, eval_act_index] = reward + self.gamma * q_next_max

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]
        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]
        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]
        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]
        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        if self.prioritized:
            _, abs_errors, self.cost = self.sess.run([self._train_op, self.abs_errors, self.loss],
                                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                                self.q_target: q_target,
                                                                self.ISWeights: ISWeights})
            self.memory.batch_update(tree_idx, abs_errors)  # update priority
        # train eval network
        else:
            _, self.cost = self.sess.run([self._train_op, self.loss],
                                         feed_dict={self.s: batch_memory[:, :self.n_features],
                                                    self.q_target: q_target})

        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self, ax_):
        # import matplotlib.pyplot as plt
        # plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        # plt.ylabel('Cost')
        # plt.xlabel('training steps')
        # plt.show()

        ax_.plot(np.arange(len(self.cost_his)), self.cost_his)
        ax_.set_title('NN_Loss_value')
        ax_.set_ylabel('Cost')
        ax_.set_xlabel('training steps')

    def fetch_target(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]  # shape=(1,n_features)

        target_values = self.sess.run(self.q_next, feed_dict={self.s_: observation})
        # max_target_value = np.max(target_values)
        return target_values

    def fetch_eval(self, observation):
        observation = observation[np.newaxis, :]

        eval_values = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        return eval_values

    def fetch_batch_sample(self):
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        return batch_memory

    def saveModel(self, node_No, step):
        #  保存模型路径
        ckpt_file_path = "./models/DQN" + '[' + str(node_No) + ']'
        path = os.path.dirname(os.path.abspath(ckpt_file_path))
        if os.path.isdir(path) is False:
            os.makedirs(path)

        tmp_path = ckpt_file_path
        self.saver.save(self.sess, tmp_path, global_step=step)

    def loadModel(self, node_No):
        ckpt_file_path = "./models/DQN" + '[' + str(node_No) + ']'
        ckpt = tf.train.get_checkpoint_state(ckpt_file_path)

        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            pass
