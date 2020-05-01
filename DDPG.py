# -*- coding: utf-8 -*-
import os
import random
import pickle
from env import ENV
from collections import deque

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Lambda, concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K


class DDPG():
    """Deep Deterministic Policy Gradient Algorithms.
    """

    def __init__(self, s_dim, a_dim, env, etd_factor):
        self.sess = K.get_session()
        self.env = env
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.etd = etd_factor
        self.bound = 1  # TODO:修改动作空间

        # update rate for target model.
        self.TAU = 0.01
        # experience replay.
        self.memory_buffer = deque(maxlen=4000)
        # discount rate for q value.
        self.gamma = 0.95
        # epsilon of action selection
        self.epsilon = 1.0
        # discount rate for epsilon.
        self.epsilon_decay = 0.995
        # min epsilon of ε-greedy.
        self.epsilon_min = 0.01

        # actor learning rate
        self.a_lr = 0.005
        # critic learining rate
        self.c_lr = 0.01

        # ddpg model
        self.actor = self._build_actor()
        # self.dense1_layer_model = Model(inputs=self.actor.input,
        #                            outputs=self.actor.get_layer('Dense_1').output)
        # self.dense2_layer_model = Model(inputs=self.actor.input,
        #                            outputs=self.actor.get_layer('Dense_2').output)
        # self.dense3_layer_model = Model(inputs=self.actor.input,
        #                           outputs=self.actor.get_layer('Dense_3').output)
        # self.dense4_layer_model = Model(inputs=self.actor.input,
        #                                 outputs=self.actor.get_layer('Dense_4').output)
        self.critic = self._build_critic()

        # target model
        self.target_actor = self._build_actor()
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic = self._build_critic()
        self.target_critic.set_weights(self.critic.get_weights())

        # gradient function

        self.get_critic_grad = self.critic_gradient()
        self.actor_optimizer()

        # if os.path.exists('model/ddpg_actor.h5') and os.path.exists('model/ddpg_critic.h5'):
        #     self.actor.load_weights('model/ddpg_actor.h5')
        #     self.critic.load_weights('model/ddpg_critic.h5')

    def _build_actor(self):
        """Actor model.
        """
        # 添加 BN
        inputs = Input(shape=(self.s_dim,), name='state_input')
        # bn_inputs = BatchNormalization(name='BN_input')(inputs)
        h1 = Dense(40, activation='relu', kernel_initializer='RandomNormal', name='Dense_1')(inputs)
        bn_h1 = BatchNormalization(name='BN_h1')(h1)
        h2 = Dense(40, activation='relu', kernel_initializer='RandomNormal', name='Dense_2')(bn_h1)
        bn_h2 = BatchNormalization(name='BN_h2')(h2)
        x = Dense(self.a_dim, activation='tanh', kernel_initializer='RandomNormal', name='Tanh_3')(bn_h2)

        # 不添加 BN
        # h1 = Dense(100, activation='relu', name='Dense_1')(inputs)
        # h2 = Dense(70, activation='relu', name='Dense_2')(h1)
        # h3 = Dense(50, activation='relu', name='Dense_3')(h2)
        # x = Dense(self.a_dim, activation='tanh', name='Dense_4')(h3)

        output = Lambda(lambda x: x * self.bound)(x)
        model = Model(inputs=inputs, outputs=output)
        # model.compile(loss='mse', optimizer=Adam(lr=self.a_lr))

        return model

    def _build_critic(self):
        """Critic model.
        """
        sinput = Input(shape=(self.s_dim,), name='state_input')
        ainput = Input(shape=(self.a_dim,), name='action_input')

        s = Dense(20, activation='relu')(sinput)
        a = Dense(20, activation='relu')(ainput)
        # x = concatenate([s, a])
        inputs = concatenate([s, a])
        h1 = Dense(40, activation='relu')(inputs)
        h2 = Dense(40, activation='relu')(h1)
        x = Dense(30, activation='relu')(h2)
        output = Dense(1, activation='linear')(x)

        model = Model(inputs=[sinput, ainput], outputs=output)
        model.compile(loss='mse', optimizer=Adam(lr=self.c_lr))

        return model

    def actor_optimizer(self):
        """actor_optimizer.

        Returns:
            function, opt function for actor.
        """
        self.ainput = self.actor.input
        aoutput = self.actor.output
        trainable_weights = self.actor.trainable_weights
        self.action_gradient = tf.placeholder(tf.float32, shape=(None, 2))

        # tf.gradients will calculate dy/dx with a initial gradients for y
        # action_gradient is dq / da, so this is dq/da * da/dparams
        self.theta_grad = tf.gradients(aoutput, trainable_weights)
        self.params_grad = tf.gradients(aoutput, trainable_weights, -self.action_gradient)
        grads = zip(self.params_grad, trainable_weights)
        self.opt = tf.train.AdamOptimizer(self.a_lr).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    def critic_gradient(self):
        """get critic gradient function.

        Returns:
            function, gradient function for critic.
        """
        cinput = self.critic.input
        coutput = self.critic.output

        # compute the gradient of the action with q value, dq/da.
        action_grads = K.gradients(coutput, cinput[1])

        return K.function([cinput[0], cinput[1]], action_grads)

    def OU(self, x, mu=0, theta=0.15, sigma=0.2):
        """Ornstein-Uhlenbeck process.
        formula：ou = θ * (μ - x) + σ * w

        Arguments:
            x: action value.
            mu: μ, mean fo values.
            theta: θ, rate the variable reverts towards to the mean.
            sigma：σ, degree of volatility of the process.

        Returns:
            OU value
        """
        return theta * (mu - x) + sigma * np.random.randn(2)

    def get_action(self, X):
        """get actor action with ou noise.

        Arguments:
            X: state value.
        """

        action = self.actor.predict(X)[0]
        # TODO:检查动作维度

        # add randomness to action selection for exploration
        noise = max(self.epsilon, 0) * self.OU(action)
        action = np.clip(action + noise, -self.bound, self.bound)
        # print("action:{}".format(action))

        return action

    def remember(self, state, action, reward, next_state, done):
        """add data to experience replay.

        Arguments:
            state: observation.
            action: action.
            reward: reward.
            next_state: next_observation.
            done: if game done.
        """
        item = (state, action, reward, next_state, done)
        self.memory_buffer.append(item)

    def update_epsilon(self):
        """update epsilon.
        """
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def process_batch(self, batch):
        """process batch data.

        Arguments:
            batch: batch size.

        Returns:
            states: states.
            actions: actions.
            y: Q_value.
        """
        y = []
        # ranchom choice batch data from experience replay.
        data = random.sample(self.memory_buffer, batch)
        states = np.array([d[0] for d in data])
        actions = np.array([d[1] for d in data])
        next_states = np.array([d[3] for d in data])

        # Q_target。
        next_actions = self.target_actor.predict(next_states)
        q = self.target_critic.predict([next_states, next_actions])

        # update Q value
        for i, (_, _, reward, _, done) in enumerate(data):
            target = reward
            if not done:
                target += self.gamma * q[i][0]
            y.append(target)

        return states, actions, y

    def update_model(self, X1, X2, y):
        """update ddpg model.

        Arguments:
            states: states.
            actions: actions.
            y: Q_value.

        Returns:
            loss: critic loss.
        """
        #        loss = self.critic.train_on_batch([X1, X2], y)
        loss = self.critic.fit([X1, X2], y, verbose=0)
        loss = np.mean(loss.history['loss'])

        X3 = self.actor.predict(X1)
        # h1_ = self.dense1_layer_model.predict(X1)
        # h2_ = self.dense2_layer_model.predict(X1)
        # h3_ = self.dense3_layer_model.predict(X1)
        # a4_ = self.dense4_layer_model.predict(X1)

        a_grads = np.array(self.get_critic_grad([X1, X3]))[0]
        self.sess.run(self.opt, feed_dict={
            self.ainput: X1,
            self.action_gradient: a_grads
        })
        # print(self.sess.run(self.params_grad, feed_dict={
        #     self.ainput: X1,
        #     self.action_gradient: a_grads
        # }))
        print(self.sess.run(self.theta_grad, feed_dict={self.ainput: X1}))
        print("*"*50)
        print("loss:{} | actions:{} ".format(loss, X3[0]))
        # print("h1:{}".format(h1_))
        # print("h2:{}".format(h2_))
        # print("h3:{}".format(h3_))
        # print("h4:{}".format(a4_))
        return loss

    def update_target_model(self):
        """soft update target model.
        formula：θ​​t ← τ * θ + (1−τ) * θt, τ << 1.
        """
        critic_weights = self.critic.get_weights()
        actor_weights = self.actor.get_weights()
        critic_target_weights = self.target_critic.get_weights()
        actor_target_weights = self.target_actor.get_weights()

        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU) * critic_target_weights[i]

        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU) * actor_target_weights[i]

        self.target_critic.set_weights(critic_target_weights)
        self.target_actor.set_weights(actor_target_weights)

    def action_process(self, action):
        # 将动作适配环境所接受的动作
        # action = [(50,1), (1,1)]
        #
        observation = self.env.observation
        node = np.argmax(observation[4:54])
        neighbors = self.env.neighbor_list[node]
        # 输出动作区间（-3,3)
        node_action = action[0]
        offload_action = action[1]
        # print("*"*50)
        # print("action[0]: {}".format(action[0]))
        # print("action[1]: {}".format(action[1]))
        # print("*"*50)
        # print("node_action:{}".format(node_action))
        # print("index：{}".format((node_action+3) * (len(neighbors)-1) / (2 * self.bound)))
        index = int((node_action + self.bound) * (len(neighbors) - 1) / (2 * self.bound))
        node_action_index = neighbors[index]

        node_action = np.zeros(len(self.env.node_list), dtype=np.int)
        node_action[node_action_index] = 1

        # res_cpt = observation[0]
        # offload_action 有两种选项，直接输出卸载的比例，或者输出卸载时间
        offload_action = (offload_action+self.bound) / (2*self.bound)       # 卸载比例
        # offload_action = (observation[0] * (offload_action + 3) / 6) / (self.env.net_states[node] + 0.01)  # 输出时间
        return [node_action, offload_action]

    def train(self, episode, batch, Iter, iter_):
        """training model.
        Arguments:
            episode: ganme episode.
            batch： batch size of episode.

        Returns:
            history: training history.
        """
        history = {'episode': [], 'Episode_reward': [], 'Loss': [], 'ec': [], 'delay': []}
        time_counter = 0
        for i in range(episode):
            observation, task_tmp = self.env.reset()
            reward_sum, ec_sum, delay_sum = self.env.init_reward(task_tmp, self.etd)
            losses = []
            flag = False
            while True:
                # if time_counter % 10 == 0:
                #     flag = True
                # else:
                #     flag = False

                # chocie action from ε-greedy.
                x = observation.reshape(-1, self.s_dim)
                # actor action
                pre_action = self.get_action(x)

                action = self.action_process(pre_action)

                reward, observation, done, ec, delay = self.env.perceive(action, self.etd, flag)
                # add data to experience replay.
                reward_sum += reward
                ec_sum += ec
                delay_sum += delay
                time_counter += 1
                self.remember(x[0], pre_action, reward, observation, done)

                if len(self.memory_buffer) > batch and time_counter % 10 == 0:
                    X1, X2, y = self.process_batch(1)

                    # update DDPG model
                    loss = self.update_model(X1, X2, y)
                    # update target model
                    self.update_target_model()
                    # reduce epsilon pure batch.
                    self.update_epsilon()

                    losses.append(loss)
                if done:
                    break

            loss = np.mean(losses)
            history['episode'].append(i)
            history['Episode_reward'].append(reward_sum)
            history['Loss'].append(loss)
            history['ec'].append(ec_sum)
            history['delay'].append(delay_sum)

            print('Episode: {}/{} {}/{} | reward: {} | loss: {:.3f} | ec: {} | delay: {} | epsilon: {}'
                  .format(i, episode, iter_, Iter, reward_sum, loss, ec_sum, delay_sum, self.epsilon))

        # self.actor.save_weights('model/ddpg_actor.h5')
        # self.critic.save_weights('model/ddpg_critic.h5')

        return history

    def evaluate(self, resBuffer):
        """play game with model.
        """
        history = {'reward': [], 'ec': [], 'delay': []}
        test_task = self.env.test_task_list
        time_counter = 0
        for i in range(len(test_task)):
            task = test_task[i]
            observation, _ = self.env.reset(task=task)
            reward_sum, ec_sum, delay_sum = self.env.init_reward(task, self.etd)
            flag = False
            while True:
                # if time_counter % 5 == 0:
                #     flag = True
                # else:
                #     flag = False
                    # chocie action from ε-greedy.
                x = observation.reshape(-1, self.s_dim)
                # actor action
                pre_action = self.get_action(x)
                action = self.action_process(pre_action)
                reward, observation, done, ec, delay = self.env.perceive(action, self.etd, flag)
                # add data to experience replay.
                reward_sum += reward
                ec_sum += ec
                delay_sum += delay
                time_counter += 1
                if done:
                    print('task: {}/{} | reward: {} | ec: {} | delay: {}'.format(i, len(test_task), reward_sum, ec_sum,
                                                                                 delay_sum))
                    break
            history['reward'].append(reward_sum)
            history['ec'].append(ec_sum)
            history['delay'].append(delay_sum)

        reward_ave = np.mean(history['reward'])
        ec_ave = np.mean(history['ec'])
        delay_ave = np.mean(history['delay'])

        resBuffer['reward'].append(reward_ave)
        resBuffer['ec'].append(ec_ave)
        resBuffer['delay'].append(delay_ave)

        # print('play...')
        # observation = self.env.reset()
        #
        # reward_sum = 0
        # random_episodes = 0
        #
        # while random_episodes < 10:
        #     self.env.render()
        #
        #     x = observation.reshape(-1, 3)
        #     action = self.actor.predict(x)[0]
        #     observation, reward, done, _ = self.env.step(action)
        #
        #     reward_sum += reward
        #
        #     if done:
        #         print("Reward for this episode was: {}".format(reward_sum))
        #         random_episodes += 1
        #         reward_sum = 0
        #         observation = self.env.reset()
        #
        # self.env.close()


if __name__ == '__main__':
    NODE_NUM = 50
    n_features = NODE_NUM * 3 + 14
    n_actions = 2
    # etd_factor = 1
    etd_factor_list = [0.1, 0.5, 2, 10]
    iteration = 100
    Iter = 10
    batch = 128
    etd_factor = 10
    if os.path.exists("topo.pk"):
        with open("topo.pk", 'rb') as f:
            env = pickle.load(f)
    else:
        env = ENV()
        with open("topo.pk", "wb") as f:
            pickle.dump(env, f)

    resBuffer = {'reward': [], 'ec': [], 'delay': []}
    model = DDPG(n_features, n_actions, env, etd_factor)
    loss = {}
    for i in range(Iter):
        history = model.train(iteration, 128, Iter, i)
        loss[str(i)] = history['Loss']
        model.evaluate(resBuffer)
    loss_his = pd.DataFrame(loss)
    loss_his.to_csv('ddpg_loss_etd_1.csv')
    res = pd.DataFrame(resBuffer)
    res.to_csv("ddpg_etd_1.csv")
    ax = res.plot(grid=True)
    fig = ax.get_figure()
    fig.savefig("ddpg_etd_1.png")


    # TODO:  调整Actor输入结构, 先concat
    # TODO:  调整卸载量输出的映射关系，修改成卸载比例
