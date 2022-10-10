
"""Estimate the number of SSED and SXOR
every point in database needs once SSED
SXOR needs to be done only when two points share the same user and the order of point in deeper layer is lager"""

def pd(x,y, z, w,id,ns,tid):
    if x==id and y>ns and z==tid:
        return True
    else:
        return w
def ns(database):
    seen={}
    database1=[]
    db=[]
    n_ssed=0
    for nn in (database):
        nn['vis']=False
        n_ssed+=len(nn)
        database1.append(nn)
        db.append(nn)
    a=len(database1)
    n_xor=0
    for i in range(a):
        for j in database1[i].index.tolist():
         if database1[i].loc[j,'vis']==True:
             continue
         else:
            database1[i].loc[j,'vis']=True
            item=database1[i].loc[j]
            uid=item.id
            ns=item.ns
            tid=item.tid
            if (uid,tid) in seen.keys():
                nsp= seen[(uid,tid)]
                for l in range(i+1, a):
                    nn1=database1[l]
                    df = nn1[(nn1.id == uid) & (nn1.ns > ns)&(nn1.ns <= nsp)]
                    ad = len(df)
                    n_xor += ad
                    for item in seen:
                        duid, dtid = item
                        if uid==duid and seen[item]<=ns and tid!= dtid:
                            ddf=nn1[(nn1.id == duid) & (nn1.ns > ns)&(nn1.tid == dtid)]
                            ad=len(ddf)
                            n_xor -= ad
                    database1[l].loc[database1[l][(database1[l].id ==uid) & (database1[l].tid ==tid)&(database1[l].ns >ns)].index.tolist(),'vis'] = True
            else:
                seen[(uid,tid)]= ns
                for l in range(i+1, a):
                    nn1=database1[l]
                    df = nn1[(nn1.id == uid) & (nn1.ns > ns)& (nn1.vis==False)]
                    ad=len(df)
                    n_xor+=ad
                    database1[l].loc[database1[l][(database1[l].id == uid) & (database1[l].tid == tid) & (database1[l].ns > ns)].index.tolist(), 'vis'] = True
    return n_ssed, n_xor

