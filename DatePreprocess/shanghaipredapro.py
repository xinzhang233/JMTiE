import feather
import pandas as pd
def readfile(filename):
    df1=pd.read_table(filename,sep=',')
    df1.columns=['uid','ts','longitude','latitude','spade','angle','guest']
    dfc = df1.loc[(df1.guest>1)]
    if len(dfc==0):
        return []
    df1['ns']=df1.index
    df1['tid']=df1.index
    df2 = df1.drop(['ts','angle'], axis=1)
    df3=df2
    df = df3.loc[((df3.longitude.diff(1)+df3.latitude.diff(1))!=0)]
    if len(df)==1:
        return []
    indexlist=[]
    for name, group in df.groupby('guest'):

        group['cns'] = group['ns'].diff(1)
        a=group.loc[(group.cns != 1)].index
        indexlist.append(a)
        b=group.loc[(group.spade==0)].index
        indexlist.append(b)
    list=[]
    for i in indexlist:
        for j in i:
            list.append(j)
    list.append(int(df.iloc[-1].ns))
    list.sort()
    for l in range(len(list)-1):
        df.tid[(df.ns>=list[l])&(df.ns<list[l+1])]=l
    df.iloc[-1,-1]=l
    df = df.drop(['spade', 'guest'], axis=1)
    return df

df_empty = pd.DataFrame(columns=['uid','longitude','latitude','ns','tid'])


import os
path = "JMTiE/DatePreprocess/Taxi_070220"
files= os.listdir(path)
for j in files:
    df=readfile(r'Taxi/'+j)
    if len(df)==0:
        os.remove(r'Taxi/'+j)
    if len(df)!=0:
        df_empty=df_empty.append(df)
feather.write_dataframe(df_empty, 'shanghai.feather')


