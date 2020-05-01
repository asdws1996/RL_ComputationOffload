from env import *
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

def lamdaOffloadMethod(env, lamda, taskNumber, tmpMax=MAX_TASK_SIMULTANEOUS):
    record = {'AveRwd': [], 'AveEc': [], 'AveDelay': [], 'AveWT': []}
    totalReward = 0
    totalEc = 0
    totalDelay = 0
    totalWT = 0
    taskFinishedNumber = 0
    while True:
        if taskFinishedNumber < taskNumber-tmpMax:
            if len(env.timeline) < tmpMax:
                seed = np.random.random()
                if seed < (tmpMax / len(env.timeline)) * 3/4:
                    env.addEvent(env.newTaskInfo(evalFlag=True))
        else:
            break
        _, tmp_task = env.getEvent()
        if tmp_task.res_cpt == 0:
            # 这里记录结束的任务
            print("task {} Coventional evaluated!| ec: {:.3f} | delay: {:.3f} | reward: {:.3f} | waiting time {:.3f}|"
                  "Timeline_len: {}".format(
                env.test_task_list.index(tmp_task.task),
                tmp_task.ec,
                tmp_task.delay,
                tmp_task.reward,
                tmp_task.WT,
                len(env.timeline)
            ))
            # 记录结果
            totalWT += tmp_task.WT
            totalReward += tmp_task.reward
            totalEc += tmp_task.ec
            totalDelay += tmp_task.delay
            taskFinishedNumber += 1
            # record['taskID']
            record['AveEc'].append(totalEc / taskFinishedNumber)
            record['AveDelay'].append(totalDelay / taskFinishedNumber)
            record['AveRwd'].append(totalReward / taskFinishedNumber)
            record['AveWT'].append(totalWT / taskFinishedNumber)
            continue
        else:
            node = tmp_task.node
            src_node = tmp_task.task[3]['src_node']
            nnodes = env.neighbor_list[src_node]
            offload_node = nnodes[np.argmax([env.net_states[i] for i in nnodes])]
            if node == src_node:
                ratioAction = 1 - lamda
            else:
                ratioAction = lamda
            nodeAction = np.zeros(NODE_NUM, dtype=int)
            nodeAction[offload_node] = 1
            action = [nodeAction, ratioAction]
            _ = env.perceive(action, tmp_task)
    res = pd.DataFrame(record)
    return res

if __name__ == '__main__':
    if os.path.exists('topo.pk'):
        with open('topo.pk', 'rb') as f:
            env = pickle.load(f)
            print("environment loaded!")
    else:
        env = ENV()
        with open("topo.pk", 'wb') as f:
            pickle.dump(env, f)

    # lamda_list = (np.array([i for i in range(11)])/10).tolist()
    # print(lamda_list)
    lamda_list = [0, 0.5, 1]
    rcd = []
    rcdRwd = []
    for i in range(len(lamda_list)):
        env.iniTimeLine(n=100, evalFlag=True)
        tmp_res = lamdaOffloadMethod(env, lamda_list[i], 1000, tmpMax=200)
        rcd.append(tmp_res)
        rcdRwd.append(tmp_res['AveRwd'])
    np_rcdRwd = np.array(rcdRwd)
    new_col = []
    for i in lamda_list:
        str_tmp = 'λ={}'.format(i)
        new_col.append(str_tmp)
    res = pd.DataFrame(np_rcdRwd.T)
    res.columns = new_col
    # res.plot()
    # plt.show()
    res.to_csv('analysis/lamda.csv')

    for i in range(len(new_col)):
        path = 'analysis/lamda_offload/lamda_{}.csv'.format(lamda_list[i])
        rcd[i].to_csv(path)