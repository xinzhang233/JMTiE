import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt, exp
import time
class Point(object):
    def __init__(self, uid='*', tid='*', sequencenumber='*', similarity=0,loc=[], counter=1):
        self.uid = uid  # id of user
        self.tid = tid  # id of trajectory
        self.ns = sequencenumber  # the order of a point in trajectory tid
        self.s = similarity  # the simialrity of the point and a query point
        self.l=loc # ['latitude', 'longitude'] of the point
        if uid == '*':
            self.counter = 0  # the number of changes of sub_trajectory
        else:
            self.counter = counter

"""compute the distance with two points"""
def geo_distance(lat1, lon1, lat2, lon2):  # km
    lon1, lat1, lon2, lat2 = map(radians, map(float, [lon1, lat1, lon2, lat2]))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r

def searp(query, database):
    T = []# record the UB_visited(counter, j)
    T_2 = []# record the tau_sub(counter,j)
    N1 = []# record knn of every query point
    for index, q in enumerate(query):
        T.append([])
        T_2.append({})
        store1=list(database[index].apply(lambda x: Point(x['id'], x['tid'], x['ns'],x['s'], [x['latitude'], x['longitude']]), axis=1))
        store1.append(Point())
        N1.append(store1)
    return N1, T, T_2

def UpB(S):#update the current maximum similarity
    for j in range(0, len(Table)):
       if Table[j][0].s>UB_t-S:
           LB_p[j]=S-(UB_t-Table[j][0].s)
           while Table[j][-1].s<LB_p[j]:
                Table[j].remove(Table[j][-1])# delete the point which similarity less than LB_p

def judge1(single_T, counter, val):#check the visited information
    if len(single_T) == 0:
        single_T.append([counter, val])
        return True
    for i in single_T:
        Tc, Tts = i[0], i[1]
        if counter >= Tc and val < Tts:  # 减少现在的
            return False
        elif counter <= Tc and Tts <= val:
            single_T.remove(i)  # 减少之前的
    single_T.append([counter, val])
    return True

def record_seen_id(Table):
    tidset_ns = {}
    tidset = {}
    for i in Table:
        for pot in i:
            if pot.uid!='*':
                if tidset.__contains__((pot.uid, pot.tid)):
                    if pot.ns not in tidset_ns[(pot.uid, pot.tid)]:
                        tidset_ns[(pot.uid, pot.tid)].add(pot.ns)
                        tidset[(pot.uid, pot.tid)].append(pot)
                else:
                    pns = set()
                    pns.add(pot.ns)
                    tidset_ns[(pot.uid, pot.tid)] = pns
                    pset = [pot]
                    tidset[(pot.uid, pot.tid)] = pset
    for k in tidset.keys():
        s=sorted(tidset[k], key=lambda x: x.ns)
        tidset[k]=s
    return tidset

def DP(query, tra):
    lena = len(query)
    lenb = len(tra)
    M = [[0 for i in range(lenb + 1)] for j in range(lena + 1)]
    flag = [[0 for i in range(lenb + 1)] for j in range(lena + 1)]
    for i in range(1,lena+1):
        for j in range(1, lenb+1):
            q = query[i-1]
            p = tra[j-1]
            dist = geo_distance(p[0], p[1], q[0], q[1])
            s=exp(-dist)
            M[i][j] = s + M[i - 1][j - 1]
            if M[i][j]< M[i][j-1]:
                M[i][j] = M[i][j - 1]
                flag[i][j] = 'left'
            elif M[i][j] <M[i-1][j]:
                M[i][j] = M[i-1][j]
                flag[i][j] = 'up'
            else:
                flag[i][j] = 'ok'
    return M,M[lena][lenb],flag

def printLcs(flag, i, j,li):
    if i == 0 or j == 0:
        return
    if flag[i][j] == 'ok':
        printLcs(flag, i - 1, j - 1,li)
        # print a[i - 1]
        li.append([i-1,j-1])
    elif flag[i][j] == 'left':
        printLcs(flag,  i, j - 1,li)
    else:
        printLcs(flag, i - 1, j,li)
    return

def single(layer, Table):
        tidset=record_seen_id(Table[layer:])
        max_lb_tra=[0,0]
        for id in tidset.keys():
            tra=[poit.l for poit in tidset[id]]
            M, LB_tra, flag = DP(query[layer:], tra)
            if LB_tra > max_lb_tra[1]:
                max_lb_tra[1] = LB_tra
                max_lb_tra[0] = id
                max_flag=flag
        id =max_lb_tra[0]
        s = max_lb_tra[1]
        loc_tra=[]
        printLcs(max_flag, len(query[layer:]), len(tidset[id]),loc_tra)
        single_path=[]
        for loc in loc_tra:
            i,j=loc
            pot=tidset[id][j]
            p=pot.l
            q=query[layer:][i]
            dist = geo_distance(p[0], p[1], q[0], q[1])
            p=Point(pot.uid, pot.tid, pot.ns, exp(-dist),pot.l)
            single_path.append(p)
        T2_single=[s,single_path]
        return T2_single

def update_T_2(layer, Table, T_2, counter, val):  #
    T_1 = [[] for _ in range(len(Table))]
    if layer >= len(Table) - (f1 - counter):
        part_tau_val = 0
        part_tau = []
        for i in range(layer, len(Table)):
            part_tau_val += Table[i][0].s
            part_tau.append(Table[i][0])
            T_1[i].append([counter + i - layer, part_tau_val])
        T_2[layer][counter] = [part_tau_val, part_tau]
    else:
        if counter not in T_2[layer].keys():
            if counter == f1 - 1:  # 2
                 T_2[layer][counter] =single(layer, Table)
            else:
                part_tau = []
                part_tau_val = 0

                for pot in Table[layer]:
                    multi_tau = [pot]
                    multi_tau_val = [pot.s]
                    funmt(layer + 1, [pot], Table, pot.s, f1 - counter, multi_tau, multi_tau_val, counter, val, T_1)
                    if multi_tau_val[0] > part_tau_val:
                        part_tau_val = multi_tau_val[0]
                        part_tau = multi_tau[0]
                lostn = len(Table) - layer - len(part_tau)
                if lostn > 0:
                    part_tau = [Point()] * lostn + part_tau
                T_2[layer][counter] = [part_tau_val, part_tau]
    return T_2[layer][counter], T_1

def update_T_1(pre_T1, T_1, counter, val, layer):
    length = len(pre_T1)
    for i in range(layer, length):
        for old_counter, old_val in T_1[i]:
            judge1(pre_T1[layer], old_counter + counter, old_val + val)
    return pre_T1

def funmt(layer, path, Table, val, f, res, res_val, inherit_counter, inherit_val, TTTT):  # 层数.路径，表格、当前的累加值
    if path[-1].counter < f and layer == len(Table):
        if val > res_val[0]:
            res[0] = path[:]
            res_val[0] = val
            UpB(val)
        return
    else:
        if judge1(TTTT[layer-1], path[-1].counter, val):
            layer_res, next_layer_T_1 = update_T_2(layer, Table, T_2, path[-1].counter + inherit_counter,
                                                   val + inherit_val)
            part_path_value, part_path = layer_res[0], layer_res[1]
            TTTT = update_T_1(TTTT, next_layer_T_1, path[-1].counter, val, layer)
            cur_val = val + part_path_value
            curpath = path + part_path
            if cur_val > res_val[0]:
                res[0] = curpath[:]
                res_val[0] = cur_val
                UpB(cur_val)
            for poi in Table[layer]:  # 只考虑当前点的FP点
                if poi.ns == '*':
                    poi.ns = path[-1].ns
                    poi.counter = path[-1].counter
                    poi.uid = path[-1].uid
                    poi.tid = path[-1].tid
                    funmt(layer + 1, path + [poi], Table, val + poi.s, f, res, res_val, inherit_counter, inherit_val,
                          TTTT)
                    poi.ns = '*'
                    poi.counter = 0
                    poi.uid = '*'
                    poi.tid = '*'
                if poi.uid == path[-1].uid and poi.tid == path[-1].tid and poi.ns > path[-1].ns:
                    poi.counter = path[-1].counter
                    funmt(layer + 1, path + [poi], Table, val + poi.s, f, res, res_val, inherit_counter, inherit_val,
                          TTTT)
                    poi.counter = 1

        else:
            for poi in Table[layer]:
                if poi.ns == '*':
                    poi.ns = path[-1].ns
                    poi.counter = path[-1].counter
                    poi.uid = path[-1].uid
                    poi.tid = path[-1].tid
                    funmt(layer + 1, path + [poi], Table, val + poi.s, f, res, res_val, inherit_counter, inherit_val,
                          TTTT)
                    poi.ns = '*'
                    poi.counter = 0
                    poi.uid = '*'
                    poi.tid = '*'
                else:
                    if poi.uid == path[-1].uid and poi.tid == path[-1].tid and poi.ns > path[-1].ns:
                        poi.counter = path[-1].counter

                        funmt(layer + 1, path + [poi], Table, val + poi.s, f, res, res_val, inherit_counter,
                              inherit_val, TTTT)
                        poi.counter = 1

    return

if __name__ == '__main__':
    total_counter=3 #the limited number of sub-trajectories
    query = np.load('query.npy')
    query = query.tolist()
    point_number = np.load('tst.npy')
    database = []
    for index, q in enumerate(query):
        db = pd.read_feather(str(index) + '.feather')
        db_xianzhidian = db.iloc[point_number - 1]['distance']
        db = db[(db.distance <= db_xianzhidian)]
        database.append(db)
    Table, T, T_2 = searp(query, database)  # 询问，个数，步长，数据库 。元素的内容[uid,tid,ns,S]

    # 4在搜索到的数据集上面做simple
    LB_p = []
    UB_t = 0
    for j in range(0, len(Table)):
        UB_t = UB_t + Table[j][0].s
        LB_p.append([])
    tds = time.perf_counter()
    f1 = total_counter
    res = [0]
    res_val = [0]
    for p in Table[0]:
        funmt(1, [p], Table, p.s, total_counter, res, res_val, 0, 0, T)

    '''output the point set'''
    print('the max similarity', res_val[0])
    print('the point set')
    for pot in res[0]:
        print(pot.uid, pot.tid, pot.ns, pot.s)
    #
    max_v = max(res_val)
    print("...........")
    for index in range(len(res)):
        if res_val[index] == max_v:
            for pot in res[index]:
                print(pot.uid, pot.tid, pot.ns, pot.s)

    # tde = time.perf_counter()
    # print('1search time:',(tde - tds))
    # print(LB_p)
    # print(T_2[5])
    from computeSS import *
    n_ssed, n_xor=ns(database)
    print('n_ssed:',n_ssed, "n_xor", n_xor)