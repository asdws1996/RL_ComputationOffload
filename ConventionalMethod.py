def RouteTableMethod(task_pool, env):
    validate_ = 0
    cost_his = []
    round_his = []
    graph = env.distance_map
    dis_ls = env.d_distance_list
    path_ls = env.path_list

    for i in range(len(task_pool)):
        task = task_pool[i]
        offload_target = task[0]
        rest_offload = task[0]
        L = task[1]
        delay_required = task[2]
        profit_ratio = task[3]
        des_node = task[4]['des_node']
        src_node = task[4]['src_node']
        present_node = src_node
        while True:
            nn = path_ls[present_node]
            # 卸载量更新
            rest_offload = rest_offload - env.net_states[present_node]
            offload_ = env.net_states[present_node] if rest_offload > env.net_states[present_node] else rest_offload
            rest_offload = rest_offload if rest_offload > 0 else 0
            # 数据量更新
            cpt_offloaded = offload_target - rest_offload
            L_ = L * (1 - (1 - profit_ratio) * (cpt_offloaded / offload_target))
            # 时延更新
            delay_offload = offload_ / self.cpt_v[node_]  # 计算时延
            delay_trans = L_ / env.trans_v[node_]
            delay_propogation = env.distance_map[present_node, nn]
            delay_ = task_['rest_delay'] - (delay_offload + delay_trans + delay_propogation)
            delay_required -= delay_
            # 能耗
            tmp_cost +=

        tmp_cost = 0
