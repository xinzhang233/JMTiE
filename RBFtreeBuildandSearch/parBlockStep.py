import time
def str2time(timezone,str, fmt='%a %b %d %H:%M:%S +0000 %Y'):#输出UTC时间
    tm = time.strptime(str, fmt)
    return time.mktime(tm)+timezone
def str2time1(timezone,str, fmt='%a %b %d %H:%M:%S +0000 %Y'):#输出UTC时间
    tm = time.strptime(str, fmt)
    tid= tm.tm_mon+tm.tm_year*100
    return tid

def ParB2R(lst,n,lamin,lamax,lomin,lomax):
    """partition the block by dichotomy
    lst: the point list in block
    n: the number of point in the range with length is one step
    lamin, lamax is the min and max in latitude
    lomin, lomax is the min and max in longitude
    v is the step length"""
    lam = round(lamin+(lamax-lamin)/2,8)
    lom= round(lomin+(lomax-lomin)/2,8)
    z=list([] for i in range(4))
    for j in lst:
        if j[0]<=lam:
           if j[1]<=lom:
              z[3].append(j)# dl=[]
           else:
              z[2].append(j)# dr=[]
        else:
            if j[1]<=lom:
              z[0].append(j)# ul = []
            else:
              z[1].append(j)# ur = []
    if len(z[3])>n:
        ParB2R(z[3], n, lamin, lam,lomin,lom)
    if len(z[2])>n:
        ParB2R(z[2], n, lamin, lam,lom, lomax)
    if len(z[0])>n:
        ParB2R(z[0], n, lam, lamax,lomin,lom)
    if len(z[1]) > n:
        ParB2R(z[1], n, lam, lamax, lom, lomax)
    else:
        if lam-lamin<v[0] and lom-lomin<v[1]:
            v[0]=round(lam-lamin,8)
            v[1]=round(lom-lomin,8)
    return

def ParDB2B(qc,num_point_block):
    """partition the database to blocks
    qc: the point in the database
    num_point_block: the largest number of points in a block
    df_lst: the dataframe list of block, one dataframe records all points in one block"""
    lamin = qc.latitude.min()
    lamax = qc.latitude.max()
    ila=[lamin]
    for j in range(1):
        mid_point = qc.quantile((j+1)*0.5)
        lamid = mid_point['latitude']
        ila.append(lamid)
    ila.append(lamax)
    for a in range(len(ila)-1):
        df_part = qc[(qc.latitude > ila[a]) & (qc.latitude < ila[a + 1])]
        if len(df_part)==0:
            continue
        else:
            if len(df_part) < num_point_block:
                df_lst.append(df_part)
            else:
                lomin = df_part.longitude.min()
                lomax = df_part.longitude.max()
                ilo = [lomin]
                for j in range(1):
                    mid_point = df_part.quantile((j + 1) * 0.5)
                    lomid = mid_point['longitude']
                    ilo.append(lomid)
                ilo.append(lomax)
                for o in range(len(ilo) - 1):
                    df_part1 = df_part[(df_part.longitude < ilo[o + 1]) & (df_part.longitude > ilo[o])]
                    if len(df_part1) == 0:
                        continue
                    else:
                        if len(df_part1) < num_point_block:
                            df_lst.append(df_part1)
                        else:
                            ParDB2B(df_part1, num_point_block)

def part(df,n,v):
    """partition the block with dataframe df
        n: the number of point in the range with length is one step
        v: the step length
        p_lst: the list of points in block
        zone: the zone of the block"""
    p_lst=list(df.apply(lambda x: [x['latitude'], x['longitude']], axis=1))
    lamin = df.latitude.min()
    lomin = df.longitude.min()
    lamax = df.latitude.max()
    lomax = df.longitude.max()
    zone=[lamin, lamax, lomin, lomax]
    ParB2R(p_lst, n, lamin, lamax, lomin, lomax)
    return p_lst,v,zone

import pandas as pd
"""read the NYK database"""
df1=pd.read_excel(r'../DatePreprocess/dataset_TSMC2014_NYC.xlsx')
df1.columns=['id','vid','vcid','vcname','latitude','longitude','timezone',"utctime"]
df1['utc'] = df1.apply(lambda x: round(str2time(x['timezone'],x['utctime']),8), axis=1)#时间转化为utc时间便于排序,此处用于判断点的顺序，做为ns
df1['tid'] = df1.apply(lambda x: round(str2time1(x['timezone'],x['utctime']),8), axis=1)
df2=df1.sort_values(by=['latitude','longitude'])#sort_index的参数axis表示排序的维度，level是排序的指标，ascending升序与否 sort_values 参数by=排序列索引
df3=df2.drop(['vid','vcid','vcname','timezone'],axis=1)
df3['latitude']=df3.latitude.round(6)
df3['longitude']=df3.longitude.round(6)
'''generate the length of step v=[lastep,lostep] for NYK database'''
lamin=df3.latitude.min()
lomin=df3.longitude.min()
lamax=df3.latitude.max()
lomax=df3.longitude.max()
qc=df3.drop_duplicates(subset=['latitude','longitude'],keep='first',inplace=False)#数据点同时具有相同的la和lo去重
n=5# the number of points in a step
v=[10,10]
L,V,Z=part(qc,n,v)
lst=[]
z_lst=[]
v_lst=[]
df_lst=[]
df_lst.append(df3)
lst.append(L)
z_lst.append(Z)
v_lst.append(v)
'''read the shanghai database'''
# df1 = pd.read_feather('../DatePreprocess/shanghai.feather')
# name=['id', 'longitude',  'latitude', 'ns', 'tid']
# df1.columns=name
# '''generate the length of step v=[lastep,lostep] for 4 blocks in shanghai database'''
# lamin = df1.latitude.min()
# lomin = df1.longitude.min()
# lamax = df1.latitude.max()
# lomax = df1.longitude.max()
# qc = df1.drop_duplicates(subset=['latitude', 'longitude'], keep='first', inplace=False)  # 数据点同时具有相同的la和lo去重
# qc=qc.sort_values(by=['latitude', 'longitude'],ascending=True)
# num_point_block=200000
# df_lst=[]
# ParDB2B(qc, num_point_block)
# lst=[]
# v_lst=[]
# z_lst=[]
# n=15# the number of points in a step
# for datafr in df_lst:
#     v = [10, 10]
#     L,V,Z=part(datafr,n,v)
#     lst.append(L)
#     z_lst.append(Z)
#     v_lst.append(v)

'''read the beijing database'''
# df1 = pd.read_feather('../DatePreprocess/beijing.feather')
# name=['latitude','longitude','ns','id','tid']
# df1.columns=name
# '''generate the length of step v=[lastep,lostep] for 64 blocks in beijing database'''
# df1 = df1[(df1.latitude>39.75) & (df1.latitude<40.01)&(df1.longitude<116.60) & (df1.longitude>116.15)]
# lamin = df1.latitude.min()
# lomin = df1.longitude.min()
# lamax = df1.latitude.max()
# lomax = df1.longitude.max()
# qc = df1.drop_duplicates(subset=['latitude', 'longitude'], keep='first', inplace=False)
# qc=qc.sort_values(by=['latitude', 'longitude'],ascending=True)
# df_lst=[]
# num_point_block=250000
# ParDB2B(qc,num_point_block)
# lst=[]
# v_lst=[]
# z_lst=[]
# n=20
# for datafr in df_lst:# we randomly choose the blocks with ids 21,24,25,38,59 in paper
#     v = [10, 10]
#     L,V,Z=part(datafr,n,v)
#     lst.append(L)
#     z_lst.append(Z)
#     v_lst.append(v)

