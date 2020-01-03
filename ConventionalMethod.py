import env

def compareNeighbors(env, node):
    neighbors = env.neighbor_list[node]
    offload_node = neighbors[0]

    preferred_v = env.net_states[offload_node] / 2 * env.distance_map[node][offload_node]
    # 选择卸载节点
    for each in neighbors[1:]:
        tmp_v = env.net_states[each] / 2 * env.distance_map[node][each]
        if preferred_v < tmp_v:
            preferred_v = tmp_v
            offload_node = each

    return offload_node

def ComparedAlgorithm(env, tasks):
    for task in tasks:
        task_offloads, rest_offloads = task[0]
        data_amount = task[1]
        pro_rate = task[2]
        des_node = task[3]['des_node']
        src_node = task[3]['src_node']
        data_pro_rate = 2e8
        # task initial

        neighbors = env.neighbor_list
        dis_dijk = env.d_distance_list
        path_dijk = env.path_list
        net_states = env.net_states
        cost_map = env.distance_map
        # 找到卸载节点
        offload_node = compareNeighbors(env, src_node)
        if net_states[offload_node] > 0:
            t_exe = rest_offloads / offload_node
            EC = env.calculate_EC(offload_node, rest_offloads)
        else:
            raise Exception("invalid topology!")

        transf_EC = env.calculate_TEC(src_node, data_amount)
        cal_data_amount = data_amount * pro_rate
        transf_EC += env.calculate_TEC(offload_node, cal_data_amount)

        t_offload = data_amount / env.trans_v[src_node]
        t_download = cal_data_amount / env.trans_v[offload_node]
        t_propogation = (data_amount+cal_data_amount) * cost_map[src_node][offload_node] / data_pro_rate

        present_node = src_node
        transf_delay = 0
        while present_node != des_node:
            transf_EC += env.calculate_TEC(present_node, cal_data_amount)
            transf_delay += cal_data_amount / env.trans_v[present_node]
            present_node = path_dijk[present_node]

        t_propogation += dis_dijk[src_node] / data_pro_rate
        t_transfer = t_download + t_offload + transf_delay

        delay_total = t_transfer + t_exe + t_propogation
        EC_total = EC + transf_EC







