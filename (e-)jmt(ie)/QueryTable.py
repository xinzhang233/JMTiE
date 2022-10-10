import feather
import random
import time
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt, exp, ceil

'''choose a trajectory in database as the query'''
def genQuery(df3, lenquery):#database, length of query
    idx = random.randint(0, len(df3))
    qp = df3.iloc[idx]
    dfq = df3[(df3.id == qp.id) & (df3.tid == qp.tid)]
    dfq = dfq.drop_duplicates(subset=['latitude', 'longitude'], keep='first', inplace=False)
    while len(dfq) < lenquery:
        idx = random.randint(0, len(df3))
        qp = df3.iloc[idx]
        dfq = df3[(df3.id == qp.id) & (df3.tid == qp.tid)]
        dfq = dfq.drop_duplicates(subset=['latitude', 'longitude'], keep='first', inplace=False)
    dfq1 = dfq.sample(lenquery, replace=False, axis=0)
    dfq1 = dfq1.sort_values(by=['ns', 'latitude', 'longitude'])
    query = list(dfq1.apply(lambda x: [x['latitude'], x['longitude']], axis=1))
    df4 = df3.drop(index=(df3.loc[(df3.id == qp.id) & (df3.tid == qp.tid)].index))#delete the trajectories whose id is the same as the query
    return query, df4

"""compute the distance with two points"""
def geo_distance(lat1, lon1, lat2, lon2):  # km
    lon1, lat1, lon2, lat2 = map(radians, map(float, [lon1, lat1, lon2, lat2]))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r

"""recording the knn points """
# def pointtoexcel(df3,query,point_number): #database, query ,k=point_number)
#     snlist=[]
#     database=[]
#     writer = pd.ExcelWriter(r'database.xlsx')
#     for index, q in enumerate(query):
#         df3['distance'] = df3.apply(lambda x: round(geo_distance(x['latitude'], x['longitude'], q[0], q[1]), 8), axis=1)
#         db=df3.sort_values(by=['distance'])
#         db_zanshi = db.iloc[point_number-1]['distance']
#         db=db[(db.distance <= db_zanshi)]
#         db['s'] = db.apply(lambda x: exp(-x['distance']), axis=1)
#         dbqc=db.drop_duplicates(subset=['latitude', 'longitude'], keep='first', inplace=False)
#         snlist.append([len(db),len(dbqc)])
#         db.to_excel(writer, sheet_name='sheet'+str(index))
#         database.append(db)
#     writer.save()
#     return database, snlist

def pointtoexcel(step_number,step_length,df3,query,point_number):
    exz = [i * step_number for i in step_length]
    snlist=[]
    database=[]
    for index, q in enumerate(query):
        db = df3[(df3.latitude<q[0]+exz[0]) & (df3.latitude>q[0]-exz[0])& (df3.longitude<q[1]+exz[1]) &(df3.longitude>q[1]-exz[1])]
        st=step_number
        while len(db)<=point_number:
            st+=step_number
            exz = [i * st for i in step_length]
            db = df3[(df3.latitude < q[0] + exz[0]) & (df3.latitude > q[0] - exz[0]) & (df3.longitude < q[1] + exz[1]) & (
                            df3.longitude > q[1] - exz[1])]
        db['distance'] = db.apply(lambda x: round(geo_distance(x['latitude'], x['longitude'], q[0], q[1]), 8), axis=1)
        db = db.sort_values(by=['distance'])
        db_zanshi = db.iloc[point_number - 1]['distance']
        db = db[(db.distance <= db_zanshi)]
        db['s'] = db.apply(lambda x: exp(-x['distance']), axis=1)
        dbqc = db.drop_duplicates(subset=['latitude', 'longitude'], keep='first', inplace=False)
        snlist.append([len(db),len(dbqc)])
        feather.write_dataframe(db, str(index)+'.feather')
        database.append(db)
    return database, snlist

"""read the NYK database"""
def str2time(timezone, str, fmt='%a %b %d %H:%M:%S +0000 %Y'):  # out put the UTC time
    tm = time.strptime(str, fmt)
    return time.mktime(tm) + timezone

def str2time1(timezone, str, fmt='%a %b %d %H:%M:%S +0000 %Y'):  #
    tm = time.strptime(str, fmt)
    tid = tm.tm_mon + tm.tm_year * 100
    return tid
df1 = pd.read_excel(r'../DatePreprocess/dataset_TSMC2014_NYC.xlsx')
df1.columns = ['id', 'vid', 'vcid', 'vcname', 'latitude', 'longitude', 'timezone', "utctime"]
df1['ns'] = df1.apply(lambda x: round(str2time(x['timezone'], x['utctime']), 8), axis=1)
df1['tid'] = df1.apply(lambda x: round(str2time1(x['timezone'], x['utctime']), 8), axis=1)
'''all the points of a user in a month consists a trajectory '''
df1 = df1.drop(['vid', 'vcid', 'vcname', 'timezone'], axis=1)
df1['latitude'] = df1.latitude.round(6)
df1['longitude'] = df1.longitude.round(6)
v=[1.3e-05, 1.8e-05]# step length in constrution of RBF tree

"""read the SHH database"""
# df1 = pd.read_feather('../DatePreprocess/shanghai.feather')
# name=[ 'id', 'longitude',  'latitude', 'ns', 'tid']
# df1.columns=name
# v=[0.00028527, 0.00027675]# step length in constrution of RBF tree
#
"""read the BJ database"""
# df1 = pd.read_feather('../DatePreprocess/beijing.feather')
# name=['latitude','longitude','ns','id','tid']
# df1.columns=name
# df = df1[(df1.latitude>39.75) & (df1.latitude<40.01)&(df1.longitude<116.60) & (df1.longitude>116.15)]
# df1 = df.round(5)
# v=[1.74e-06, 2.65e-06]# step length in constrution of RBF tree


len_query=8# the length of query
query, df4 = genQuery(df1,len_query)
point_number=50# the parameter k of JMT and e-JMT
step_number=ceil(point_number)#step_number=ceil(point_number/15) in SHH and BJ datasets
result=pointtoexcel(step_number,v,df4,query,point_number)
database,l=result
np_query = np.array(query)
np.save('query.npy', np_query)
np_tst = np.array(point_number)
np.save('tst.npy', np_tst)
