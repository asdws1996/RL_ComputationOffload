"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
Using:
tensorflow 1.0

"""

import tensorflow as tf
import numpy as np
from onehot import *
import time
import os

#####################  hyper parameters  ####################

# MAX_EPISODES = 200
# MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32


###############################  DDPG  ####################################


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound, env):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)

        self.pointer = 0
        self.sess = tf.Session()
        self.env = env
        self.node_num = len(env.node_list)


        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's1')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's1_')
        self.S2 = tf.placeholder(tf.float32, [None, s_dim+self.node_num], 's2')
        self.S2_ = tf.placeholder(tf.float32, [None, s_dim+self.node_num], 's2_')
        self.A1 = tf.placeholder(tf.float32, [None, 1], 'action1')
        self.A1_ = tf.placeholder(tf.float32, [None, 1], 'action1_')
        self.A1_prob = tf.placeholder(tf.float32, [None, 1], 'action1_prob')
        self.A2 = tf.placeholder(tf.float32, [None, 1], 'action2')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor1'):
            self.a1 = self._build_a1(self.S, scope='eval', trainable=True)
            self.a1_ = self._build_a1(self.S_, scope='target', trainable=False)

        with tf.variable_scope('Actor2'):
            self.a2 = self._build_a2(self.S2, scope='eval', trainable=True)
            self.a2_ = self._build_a2(self.S2_, scope='target', trainable=False)

        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor

            q = self._build_c(self.S, self.A1, self.a2, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, self.A1_, self.a2_, scope='target', trainable=False)

        # networks parameters
        self.a1e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor1/eval')
        self.a1t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor1/target')
        self.a2e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor2/eval')
        self.a2t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor2/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e) for t, e in
                             zip(self.a1t_params + self.a2t_params + self.ct_params,
                                 self.a1e_params + self.a2e_params + self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        with tf.variable_scope('Critic/train'):
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        # log_prob = tf.log(self.A1_prob)
        log_prob = tf.log(self.a1[0, self.A1])
        a1_loss = - tf.reduce_mean(log_prob * q)
        with tf.variable_scope('Actor1/train'):
            self.a1train = tf.train.AdamOptimizer(LR_A).minimize(a1_loss, var_list=self.a1e_params)

        a2_loss = - tf.reduce_mean(q)    # maximize the q
        with tf.variable_scope('Actor2/train'):
            self.a2train = tf.train.AdamOptimizer(LR_A).minimize(a2_loss, var_list=self.a2e_params)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=1)

    def choose_action(self, s):
        neighbor_list = self.env.neighbor_list
        n_ = one_hot_decode(s[4:54])
        action_limits = neighbor_list[n_]

        s = s[np.newaxis, :]
        a1_list = self.sess.run(self.a1, {self.S: s})
        a1_ald = np.array([a1_list[0, index] for index in action_limits])
        if 0 in a1_ald:
            a1 = np.random.choice(action_limits)
        else:
            a1_ald = a1_ald / a1_ald.sum()
            a1 = np.random.choice(action_limits, p=a1_ald)
        a1 = one_hot_code(self.node_num, a1)
        a1 = a1[np.newaxis, :]
        s2 = np.hstack((s, a1))
        a2 = self.sess.run(self.a2, {self.S2: s2})
        return a1[0], a2[0]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)
        if self.pointer > MEMORY_CAPACITY:
            indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        else:
            indices = np.random.choice(range(self.pointer), size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        ba1 = bt[:, self.s_dim: self.s_dim + self.a_dim - 1]
        ba2 = bt[:, -self.s_dim - 2: -self.s_dim - 1]
        # 将批量的ba1进行 one-hot编码
        ba1_0 = ba1.ravel()
        z = np.zeros([BATCH_SIZE, self.node_num])
        for i in range(BATCH_SIZE):
            z[i, ba1_0[i]] = 1
        bs2 = np.hstack((bs, z))

        # Actor 1 training
        act_FOR_A1 = self.sess.run(self.a2_, {self.S2: bs2})
        self.sess.run(self.a1train, {self.S: bs, self.A1: ba1, self.a2: act_FOR_A1})

        # Actor 2 training
        act_probs_FOR_A2 = self.sess.run(self.a1_, {self.S_: bs})
        # 对动作选取加限制
        act_FOR_A2_tmp = []
        for index in range(BATCH_SIZE):
            node_tmp = one_hot_decode(bs[index][4:54])
            action_limit = self.env.neighbor_list[node_tmp]
            tmp_ = np.array([act_probs_FOR_A2[each] for each in action_limit])
            if 0 in tmp_:
                a1_tmp = np.random.choice(action_limit)
            else:
                tmp_ = tmp_ / tmp_.sum()
                a1_tmp = np.random.choice(action_limit, p=tmp_)
            act_FOR_A2_tmp.append(a1_tmp)
        act_FOR_A2 = np.array(act_FOR_A2_tmp).ravel()
        z = np.zeros([BATCH_SIZE, self.node_num])
        for i in range(BATCH_SIZE):
            z[i, act_FOR_A2[i]] = 1
        bs2 = np.hstack((bs, z))
        self.sess.run(self.a2train, {self.S2: bs2})

        # Critic training
        # a1_probs = self.sess.run(self.a1_, {self.S_: bs_})
        # act1_c_tmp = []
        # for index in range(bs_.shape[0]):
        #     node_tmp = one_hot_decode(bs_[index][4:54])
        #     action_limit = self.env.neighbor_list[node_tmp]
        #     tmp_ = np.array([a1_probs[each] for each in action_limit])
        #     tmp_ = tmp_ / tmp_.sum()
        #     a1_tmp = np.random.choice(action_limit, p=tmp_)
        #     a1_tmp = one_hot_code(len(self.env.node_list), a1_tmp)
        #     act1_c_tmp.append(a1_tmp)
        # act1_c = np.array(act_FOR_A2_tmp)
        act_probs_FOR_C = self.sess.run(self.a1_, {self.S_: bs_})
        # 对动作选取加限制
        act_FOR_C_tmp = []
        for index in range(BATCH_SIZE):
            node_tmp = one_hot_decode(bs[index][4:54])
            action_limit = self.env.neighbor_list[node_tmp]
            tmp_ = np.array([act_probs_FOR_C[each] for each in action_limit])
            if 0 in tmp_:
                a1_tmp = np.random.choice(action_limit)
            else:
                tmp_ = tmp_ / tmp_.sum()
                a1_tmp = np.random.choice(action_limit, p=tmp_)
            act_FOR_C_tmp.append(a1_tmp)
        act_FOR_C = np.array(act_FOR_A2_tmp).ravel()
        z = np.zeros([BATCH_SIZE, self.node_num])
        for i in range(BATCH_SIZE):
            z[i, act_FOR_C[i]] = 1
        bs2_ = np.hstack((bs_, z))
        act_FOR_C = act_FOR_C[:, np.newaxis]
        self.sess.run(self.ctrain, {self.S: bs, self.A1: ba1, self.a2: ba2, self.R: br,
                                    self.S_: bs_, self.A1_: act_FOR_C, self.S2_: bs2_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a2(self, s2, scope, trainable):       # s2 应该包括前一个动作 dim_s2 = dim_s1 + one_hot_code下的a1(node_num)
        with tf.variable_scope(scope):
            # s = s[np.newaxis, :]
            # a = a[np.newaxis, :]
            # s2 = np.hstack((s, a))
            # s2 = tf.concat([s, a], 1)
            net = tf.layers.dense(s2, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a2 = tf.layers.dense(net, 1, activation=tf.nn.tanh, name='a2', trainable=trainable)
            return tf.divide(tf.multiply(a2+1, self.a_bound, name='scaled_a2'), 2)

    def _build_a1(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            act_prob = tf.layers.dense(net, self.node_num, activation=tf.nn.softmax, trainable=trainable)
            # a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            # return np.random.choice(np.arange(act_prob.shape[1]), p=act_prob.ravel())
            return act_prob


    def _build_c(self, s, a1, a2, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            # a2 = a2[:, np.newaxis]
            # a = np.hstack((a1, a2))
            a1 = one_hot_code(self.node_num, a1)
            w1_a1 = tf.get_variable('w1_a1', [self.node_num, n_l1], trainable=trainable)
            w1_a2 = tf.get_variable('w1_a2', [1, n_l1], trainable=trainable)
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a1, w1_a1) + tf.matmul(a2, w1_a2) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

    def saveModel(self, step):
        ckpt_file_path = "./models/DQN.ckpt"
        path = os.path.dirname(os.path.abspath(ckpt_file_path))

        if os.path.isdir(path) is False:
            os.makedirs(path)
        self.saver.save(self.sess, ckpt_file_path, global_step=step)
