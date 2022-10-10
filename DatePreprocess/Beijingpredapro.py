import feather
'''
transform plt to txt
'''
import os
path = "JMTiE/DatePreprocess/Geolife Trajectories 1.3/Data"
files= os.listdir(path)
for j in files:
    subfiles=[]
    subpath='Data/'+j+"/Trajectory"
    subfiles=os.listdir(subpath)
    for i in subfiles:
        fp1 = open(r'JMTiE/DatePreprocess/BeiJingdatabase/Data/'+j+"/Trajectory/"+i)
        lines = fp1.readlines()
        fp1.close()
        l_list = lines[6:]
        fw = 'JMTiE/DatePreprocess/BeiJingdatabase/Datatxt/'+j+i+'.txt'
        fp = open(fw, 'w+')
        for line in l_list:
            fp.write(line)
        fp.close()


"""read text to beijing.feather，beijing1.feather，beijing2.feather中"""
import pandas as pd
def readfile(filename):
    df1=pd.read_table(filename,sep=',')
    df1.columns=['latitude','longitude','zero','altitude','data','data_string','time_string']
    df1['ns']=df1.index
    df = df1.drop(['zero','altitude','data','data_string','time_string'], axis=1)
    return df

df_empty = pd.DataFrame(columns=['latitude','longitude','ns','uid','tid'])
import os

path = "JMTiE/DatePreprocess/BeiJingdatabase/Datatxt"
files= os.listdir(path)
for j in files[0:6000]:
    df=readfile(r'Datatxt/'+j)
    df['uid'] = j[0:3]
    df['tid'] = j[3:11]
    df_empty=df_empty.append(df)

feather.write_dataframe(df_empty, 'beijing0.feather')
df_empty = pd.DataFrame(columns=['latitude','longitude','ns','uid','tid'])
for j in files[6000:12000]:
    # print(j)
    df=readfile(r'Datatxt/'+j)
    df['uid'] = j[0:3]
    df['tid'] = j[3:11]
    df_empty=df_empty.append(df)

feather.write_dataframe(df_empty, 'beijing1.feather')
df_empty = pd.DataFrame(columns=['latitude','longitude','ns','uid','tid'])
for j in files[12000:]:
    # print(j)
    df=readfile(r'Datatxt/'+j)
    df['uid'] = j[0:3]
    df['tid'] = j[3:11]
    df_empty=df_empty.append(df)

feather.write_dataframe(df_empty, 'beijing2.feather')
df_empty = pd.DataFrame(columns=['latitude','longitude','ns','uid','tid'])
for i in range(3):
    df=pd.read_feather('beijing'+str(i)+'.feather')
    print(len(df))
    df_empty=df_empty.append(df)
print(len(df_empty))
feather.write_dataframe(df_empty, 'beijing.feather')
print("ok")