import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt, exp, ceil
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
def geo_distance(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, map(float, [lon1, lat1, lon2, lat2]))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r

def searp(query, database):
    T = []
    T_2 = []
    N1 = []
    for index, q in enumerate(query):
        T.append([])
        T_2.append({})
        store1=list(database[index].apply(lambda x: Point(x['id'], x['tid'], x['ns'], [x['latitude'], x['longitude']]), axis=1))
        N1.append(store1)
    return N1, T, T_2

def record_seen_id(Table):
    tidset_ns={}
    tidset={}
    for i in Table:
        for pot in i:
            if tidset.__contains__((pot.uid,pot.tid)):
               if pot.ns not in tidset_ns[(pot.uid,pot.tid)]:
                   tidset_ns[(pot.uid, pot.tid)].add(pot.ns)
                   tidset[(pot.uid, pot.tid)].append(pot)

            else:
                pns=set()
                pns.add(pot.ns)
                tidset_ns[(pot.uid, pot.tid)]=pns
                pset=[pot]
                tidset[(pot.uid,pot.tid)]=pset
    for k in tidset.keys():
        # tidset[k]=sorted(tidset[k], key=lambda x: x.ns)
        s=sorted(tidset[k], key=lambda x: x.ns)
        tidset[k]=[pot.s for pot in s]
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
        li.append(i-1)
    elif flag[i][j] == 'left':
        printLcs(flag,  i, j - 1,li)
    else:
        printLcs(flag, i - 1, j,li)



if __name__ == '__main__':
    query = np.load('query.npy')
    query = query.tolist()
    point_number = np.load('tst.npy')
    database = []
    for index, q in enumerate(query):
        db = pd.read_feather(str(index) + '.feather')
        db_xianzhidian = db.iloc[point_number - 1]['distance']
        db = db[(db.distance <= db_xianzhidian)]
        database.append(db)
    tds = time.perf_counter()
    # shuzhiTable, Table, N_step, unseen_UB, last_p,LB_p = init_table(query, total_step_number, v, 1, database)
    Table,  T, T_2 = searp(query, database)
    max_lb_tra = [0,0]
    tidset = record_seen_id(Table)
    for id in tidset.keys():
        M, LB_tra, flag = DP(query, tidset[id])
        if LB_tra > max_lb_tra[1]:
            max_lb_tra[1] = LB_tra
            max_lb_tra[0] = id
    print(max_lb_tra)
    tde = time.perf_counter()
    print((tde - tds))
    from computeSS import *
    n_ssed, n_xor=ns(database)
    print('n_ssed:',n_ssed, "n_xor", n_xor)

