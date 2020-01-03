import numpy as np
import random

def dijkstra(graph, dest):
    max_value = 10000
    lenth = len(graph)
    node_ls = [i for i in range(lenth)]

    Path = [-1 for _ in node_ls]
    Path[dest] = dest
    for each in node_ls:
        if graph[each][dest] != 0:
            Path[each] = dest

    S = [dest]
    V = [each for each in node_ls]
    V.remove(dest)
    s_distance = [0 for _ in node_ls]
    v_distance = [graph[dest][each] for each in V]
    for index in range(len(V)):
        if v_distance[index] == 0:
            v_distance[index] = max_value

    while V:
        nn_dis = min(v_distance)
        nn = V[v_distance.index(nn_dis)]
        S.append(nn)
        s_distance[nn] = nn_dis
        V.remove(nn)
        v_distance.remove(nn_dis)
        # 更新距离
        if V:
            for j in range(len(v_distance)):
                res_nn = V[j]
                dis_tmp = max_value if graph[nn][res_nn] == 0 else graph[nn][res_nn]
                if v_distance[j] > nn_dis + dis_tmp:
                    v_distance[j] = nn_dis + dis_tmp
                    Path[res_nn] = nn

    return s_distance, Path


def create_topology(Node_Num=50, edge_prob=0.1):
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

    # 记录拓扑图
    print("Topology Created!")
    distance_map = np.copy(M)
    for i in range(Node_Num):
        for j in range(i + 1, 50):
            if distance_map[i, j] == 1:
                distance_map[i, j] = random.randint(20, 50)
                distance_map[j, i] = distance_map[i, j]
    print(distance_map)
    return distance_map


if __name__ == '__main__':
    graph = create_topology()
    s_dis, path = dijkstra(graph, 49)
    print("the last hop")
    print(path)
    print("the distance")
    print(s_dis)