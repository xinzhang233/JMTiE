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

def init_table(query, database,point_number):
    N_pn=[]
    Table = []
    shuzhiTable = []
    T=[]
    sum_layer = []
    sum_cum_layer = 0
    for index, q in enumerate(query):
        T.append([])
        pn_zanshi=ceil(point_number/2)
        df=database[index]
        db_zanshi = df.iloc[pn_zanshi - 1]['distance']
        db = df[(df.distance <= db_zanshi)]
        store1 = list(db.apply(lambda x: Point(x['id'], x['tid'], x['ns'], x['s'], [x['latitude'], x['longitude']]),axis=1))
        sum_layer.append(store1[-1].s)
        store1.append(Point())
        Table.append(store1)
        N_pn.append(len(db))
    sum_layer.reverse()
    unseen_UB = []
    for i in sum_layer:
        sum_cum_layer = sum_cum_layer + i
        unseen_UB.append(sum_cum_layer)
    unseen_UB.reverse()
    unseen_UB[-1] = False
    return shuzhiTable, Table, N_pn,T,unseen_UB

def init(total_counter):
    f = total_counter
    f1=total_counter
    T_2=[{f1 - 1: [False]} for _ in range(len(Table))]
    tidset = {(Table[-1][0].uid, Table[-1][0].tid)}
    T_2[-1][f1-1] = [True, Table[-1][0].s, [Table[-1][0]], tidset]
    res = [0]
    res_val = [0]
    LB_p = []
    UB_t = 0
    for j in range(0, len(Table)):
        UB_t = UB_t + Table[j][0].s
        LB_p.append([True,-1])
    return f1,f, res, res_val, LB_p, UB_t,T_2,


def judge1(single_T, counter, val):
    if len(single_T) == 0:
        single_T.append([counter, val])
        return True
    for i in single_T:
        Tc, Tts = i[0], i[1]
        if counter >= Tc and val < Tts:
            return False
        elif counter <= Tc and Tts <= val:
            single_T.remove(i)
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
        return T2_single,tidset

def functauub(layer, uid,tid, Table, ub):
    for i in range(layer,len(Table)):
        s=0
        for pot in Table[i]:
            if pot.uid == uid and pot.tid == tid:
                s=pot.s
                break
            else:
                    continue
        if s==0:
            if pot.s == 0:
                s = Table[i][-2].s
            else:
                s = pot.s
        ub=ub+s
    return ub

def updatesingle_T_2(layer,Table, T_2, unseen_UB):
    if layer==len(Table)-1:
        return T_2[layer][f1-1]
    else:
        if not T_2[layer][f1-1][0]:
            if unseen_UB[layer]:
                single_res,tidset= single(layer, Table)
                part_res_val, part_res=single_res
                if unseen_UB[layer] < part_res_val:
                    unseen_UB[layer] = False
                else:
                    tidset=[]
            else:#需要查看T_2中存在的轨迹id
                part_res_val =T_2[layer][f1-1][1]
                part_res = T_2[layer][f1-1][2]
                seensetlist = []
                tidset =T_2[layer][f1-1][3]
                currentset = {}
                for item in tidset.keys():
                    (uid,tid)=item
                    poit_ub = functauub(layer,uid,tid,Table, 0)
                    if poit_ub > part_res_val:
                        currentset[item]=tidset[item]
                        seensetlist.append([item, poit_ub])
                tidset=currentset
                if len(tidset)>0:
                    seensetlist.sort(key=lambda x:x[1])
                    max_lb_tra=0
                    for i in seensetlist:
                        id,ub=i
                        tra = [poit.l for poit in tidset[id]]
                        M, LB_tra, flag = DP(query[layer:], tra)
                        if LB_tra > max_lb_tra:
                            max_lb_tra = LB_tra
                            max_lb_id = id
                            max_flag = flag
                        else:
                            break
                    id = max_lb_id
                    s = max_lb_tra
                    loc_tra = []
                    printLcs(max_flag, len(query[layer:]), len(tidset[id]), loc_tra)
                    single_path = []
                    for loc in loc_tra:
                        i, j = loc
                        pot = tidset[id][j]
                        p = pot.l
                        q = query[layer:][i]
                        dist = geo_distance(p[0], p[1], q[0], q[1])
                        p = Point(pot.uid, pot.tid, pot.ns, exp(-dist), pot.l)
                        single_path.append(p)
                    if s>part_res_val:
                        part_res_val=s
                        part_res=single_path
            T_2[layer][f1-1] = [True, part_res_val, part_res, tidset]
        return T_2[layer][f1-1]

def update_T_2(layer, Table, T_2, counter, val):
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
            part_tau = []
            part_tau_val = 0

            for pot in Table[layer]:
                multi_tau = [pot]
                multi_tau_val = [pot.s]
                funmt(layer + 1, [pot], Table, pot.s, f1 - counter, multi_tau, multi_tau_val, counter, val, T_1)
                if multi_tau_val[0] > part_tau_val:
                    part_tau_val = multi_tau_val[0]
                    part_tau = multi_tau[0]
            # if type(part_tau) == Point:
            #     part_tau = [part_tau]
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
            UpB(val, LB_p)
        return
    else:
        if judge1(TTTT[layer-1], path[-1].counter, val):
            if path[-1].counter ==f-1:
                _, part_path_value, part_path,_ =updatesingle_T_2(layer, Table, T_2, unseen_UB)
            else:
                layer_res, next_layer_T_1 = update_T_2(layer, Table, T_2, path[-1].counter + inherit_counter,
                                                       val + inherit_val)
                part_path_value, part_path = layer_res[0], layer_res[1]
                TTTT = update_T_1(TTTT, next_layer_T_1, path[-1].counter, val, layer)
            cur_val = val + part_path_value
            curpath = path + part_path
            if cur_val > res_val[0]:
                res[0] = curpath[:]
                res_val[0] = cur_val
                UpB(cur_val, LB_p)
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
                    # if judge1(T[layer],path[-1].counter,val+pot.s):
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


def output(res, res_val):
    max_v = res_val[0]
    print('the max similarity', max_v)
    print('the point set')
    for pot in res[0]:
        print(pot.uid, pot.tid, pot.ns, pot.s)
    return max_v


def UpB(S, LB_p):
    for j in range(len(Table)):
        if len(Table[j]) == 0:
            break
        if Table[j][0].s > UB_t - S:
            LB = S - (UB_t - Table[j][0].s)
            # if LB>LB_p[j][1]:
            LB_p[j][1]=LB
            if Table[j][-1].uid == "*":
                Table[j]=Table[j][:len(Table)-1]
            if Table[j][-1].s<LB:
                    for index, pot in enumerate(Table[j]):
                        if pot.s < LB:
                                Table[j] = Table[j][:index]
                                break
                    LB_p[j][0] = False
    return LB_p


def expand_table(Table, LB_p, N_pn, database,unseen_UB,T_2):
    sum_layer = []
    sum_cum_layer = 0
    T_2_new=[]
    for index, flag in enumerate(LB_p):
        T_2_single=T_2[index][f1-1]
        T_2_single[0] = False
        T_2_new.append({f1-1:T_2_single})
        if flag[0]:
            df = database[index]
            pn_zanshi = N_pn[index] + ceil((len(df) - N_pn[index]) / 2)
            N_pn[index] = pn_zanshi
            db_zanshi = df.iloc[pn_zanshi - 1]['distance']
            db = df[(df.distance <= db_zanshi)]
            store = list(db.apply(
                lambda x: Point(x['id'], x['tid'], x['ns'], x['s'], [x['latitude'], x['longitude']]), axis=1))
            fg=0
            if len(store)==len(df):
                fg=1
            # print(store[-1].s)
            if store[-1].s<flag[1]:
                    for indx, pot in enumerate(store):
                        if pot.s < flag[1]:
                                store = store[:indx]
                                LB_p[index][0] = False
                                break
            if flag[1]<=0:
                store.append(Point())
            Table[index] = store
            if fg==1:
                LB_p[index][0] = False
                print('{0}/{1}layer expanding finished'.format(index, len(Table)))
    T_2_new[-1][f-1][0] = True
    for layer in Table:
        if layer[-1].s>0:
            sum_layer.append(layer[-1].s)
        else:
            sum_layer.append(layer[-2].s)
    sum_layer.reverse()
    for index,i in enumerate(sum_layer):
        sum_cum_layer = sum_cum_layer + i
        if unseen_UB[len(sum_layer)-1-index]:
            unseen_UB[len(sum_layer)-1-index]=sum_cum_layer
    # print(N_step)
    return Table, N_pn,T_2_new,unseen_UB


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
    total_counter=3
    # k=1
    shuzhiTable, Table, N_pn,T,unseen_UB = init_table(query,database,point_number)
    f1,f, res, res_val, LB_p, UB_t,T_2, = init(total_counter)  # Ttau,

    stop = False
    while not stop:
        for p in Table[0]:
            funmt(1, [p], Table, p.s, f, res, res_val, 0, 0, T)
        # output(res, res_val)

        LB_p = UpB(res_val[0], LB_p)
        # print('LB_p:', LB_p)

        flag = False
        for bo in LB_p:
            if bo[0]:
                flag = True
        if not flag:
            stop = True
        if not stop:
            f=total_counter
            f1=total_counter
            Table, N_pn,T_2,unseen_UB = expand_table(Table, LB_p, N_pn, database,unseen_UB,T_2)

    output(res, res_val)
    # tde = time.perf_counter()
    # print('2search time:',(tde - tds))
    # print('N_step:', N_pn)
    # print('LB_p:', LB_p)
    db_real=[]
    for index, step in enumerate(N_pn):
        df = database[index]
        pn_zanshi=N_pn[index]
        db_zanshi = df.iloc[pn_zanshi - 1]['distance']
        dbr = df[(df.distance <= db_zanshi)]
        db_real.append(dbr)
    from computeSS import *
    n_ssed, n_xor = ns(db_real)
    print('n_ssed:', n_ssed, "n_xor", n_xor)

