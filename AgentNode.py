from RL import DeepQNetwork
import numpy as np

class Agent:
    def __init__(self):
        # self.neighbors = neighbors
        n_action = 50
        n_feature = 56  # s = { t(RC,TC,RT,TT, lamda), No., netStates[...] }
        self.DQN = DeepQNetwork(n_action, n_feature)
