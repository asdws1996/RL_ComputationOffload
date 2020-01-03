import numpy as np


def one_hot_code(num, target_num):
    result = np.zeros(num)
    result[target_num] = 1
    return result


def one_hot_decode(hotcode):
    return int(np.argwhere(hotcode == 1))
