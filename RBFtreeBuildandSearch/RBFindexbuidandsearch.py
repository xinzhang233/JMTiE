from parBlockStep import *
import numpy as np
import mmh3  # generate Hash function
import hashlib
import time
import random, math
"""parameter to generate bloom filter
num_bit: the number of bits in a bloom filter
num_hash: the number of hash mapped to a bloom filter"""
num_bit=1000
num_hash=7
def sha256hex(data):  # type of input is str
 sha256 = hashlib.sha256()
 sha256.update(data.encode())
 res = sha256.hexdigest()  #
 resm2 = int(res, 16) % 2
 return resm2

class BloomFilter(set):
 def __init__(self, size, hash_count, rk):  # the bits in a Bloomfilter，the number of hash functions，random number rk
  super(BloomFilter, self).__init__()
  ar = np.random.randint(0, 2, size=size)
  inar = np.invert(ar) + 2
  self.bit_array = np.array((ar, inar))
  self.size = size
  self.hash_count = hash_count
  self.rk = rk

 def __len__(self):
  return self.size

 def __iter__(self):
  return iter(self.bit_array)

 def add(self, phkw):
  for index in phkw:
   column = sha256hex(str(np.bitwise_xor(mmh3.hash(str(index), self.hash_count), self.rk)))
   self.bit_array[column][index] = 1
   self.bit_array[1 - column][index] = 0

  return self

 def __contains__(self, qhkkw):
  out = False
  for item in qhkkw[0]:
   flag = 0
   for index in item:
    column = sha256hex(str(np.bitwise_xor(mmh3.hash(str(index), self.hash_count), self.rk)))
    if self.bit_array[column][index] == 0:
     flag = 1
     break
   if flag == 1:
    continue
   else:
    for item in qhkkw[1]:
     flag1 = 0
     for index in item:
      column = sha256hex(str(np.bitwise_xor(mmh3.hash(str(index), self.hash_count), self.rk)))
      if self.bit_array[column][index] == 0:
       flag1 = 1
       break
     if flag1 == 1:
      continue
     else:
      out = True
  return out

class RBLn(object): #RBLtree node, two kinds: leafnode: child is the indexes of the points, non-leaf node: child is the child nodes
 def __init__(self, rk, child=[], phkkw=[]):
 # 设置属性
  self.lidx=phkkw
  self.bl=BloomFilter(1000,7,rk)
  # In beijing database and shanghai database: self.bl=BloomFilter(10000,10,rk)
  self.child=child
  if type(self.child[0])==RBLn:
   for ch in self.child:
    self.lidx=list(set(self.lidx) | set(ch.lidx))
  self.bl.add(self.lidx)

def PF(idx, h):
    m=bin(idx)[2:]
    ml = m.zfill(h-2)
    di=[ml]
    if idx>0:
        for l in range(0, len(m)):
            mpf=m[0:len(m)-l-1]+''.join(str('*') for _ in range(l+1))
            mpfl=mpf.zfill(h-2)
            di.append(mpfl)
    return di

def phkkw(lakey, lokey, hash_count, size):
        index = []
        for seed in range(hash_count):
            for keyword in lokey:
                item = (str('lo') + keyword)
                hkw = mmh3.hash(item, seed) % size
                index.append(hkw)
            for keyword in lakey:
                item = (str('la') + keyword)
                hkw = mmh3.hash(item, seed) % size
                index.append(hkw)

        return index
def qhkkw(larange, lorange, size, hash_count):#
    index = [[],[]]
    for i in range(0,len(larange)):
        index[0].append([])
        item = (str('la') + larange[i])
        for seed in range(hash_count):
            hkw = mmh3.hash(item, seed) % size
            index[0][i].append(hkw)
    for j in range(0,len(lorange)):
        index[1].append([])
        item = (str('lo') + lorange[j])
        for seed in range(hash_count):
            hkw = mmh3.hash(item, seed) % size
            index[1][j].append(hkw)
    return index

def buildtree(df3,v,lst,n, z):#build RBFtree
    lamin, lamax, lomin, lomax=z
    """point mapped to integer"""
    lenla= len(bin(math.floor((lamax-lamin)/v[0])))
    lenlo=len(bin(math.floor((lomax-lomin)/v[1])))
    leafnode=[]
    for i in lst:
        i0=math.floor((i[0]-lamin)/ v[0])
        lakey=PF(i0, lenla)
        i1=math.floor((i[1]-lomin)/v[1])
        lokey=PF(i1, lenlo)
        phkw=phkkw(lakey, lokey,7,1000)
        leafnode.append(RBLn(lst.index(i),df3[(df3.latitude == i[0])& (df3.longitude == i[1])].index.tolist(),phkw))
    """construct RBFtree"""
    root=[]

    def RBLtree(leafnode, n, rk=len(leafnode)):  # initial rk is the len(leafnode), n is the max number of child nodes(points) in one father node
        l = len(leafnode)
        layer = []
        i = 0
        while i < math.ceil(l / n):
            if i * n + n > l:
                child = leafnode[i * n:]
            else:
                child = leafnode[i * n:(i * n + n)]
            layer.append(RBLn(rk + i, child))
            i = i + 1
        if len(layer) == 1:
            root.append(layer[0])
        else:
            RBLtree(layer, n, rk + i)
    RBLtree(leafnode, n)
    return  root
"""building tree finished """

""" generate the PF set for the search range"""
def lnz(x):#find the last number that is not zero
    if x==0:
        return 0
    else:
        for i in range(0,len(bin(x))):
            if x%(2**i)==0:
                continue
            else:
                return i

def PFr(zone, h,key):#zone=[a,b] search region b>=x>=a, h: the number of bits for the max value（including 0b）
    if zone[0]==zone[1]:
        b=bin(zone[0])[2:]
        key.append(b.zfill(h-2))
        return
    else:
        i=lnz(zone[0])
        if i<=1:
           b=bin(zone[0])[2:]
           key.append(b.zfill(h-2))
           zone[0]=zone[0]+1
           PFr(zone, h,key)
        else:
            le=len(bin(zone[0]))
            r=zone[1]-zone[0]
            delta=2**(i-1)-1
            if r>=delta:
                b = bin(zone[0])[2:(le - (i-1))] + ''.join(str('*') for _ in range(i-1))
            else:
                j = len(bin(r)) - 2
                if j==1:
                    b = bin(zone[0])[2:(le - 1)] + ''.join(str('*') for _ in range(1))
                    delta=1
                else:
                    delta=2**(j-1)-1
                    b = bin(zone[0])[2:(le - (j-1))] + ''.join(str('*') for _ in range(j-1))
            key.append(b.zfill(h - 2))
            ub = zone[0] + delta
            if ub==zone[1]:
                return
            else:
                PFr([ub+1, zone[1]], h,key)


def range2hkkw(larange,lorange, lenla,lenlo):#[larangemin,larangemax],[lorangemin,lorangemax]
    if larange[0]<lamin:
        larange[0]=lamin
    if larange[1]>lamax:
        larange[1]=lamax
    if lorange[0]<lomin:
        lorange[0]=lomin
    if lorange[1]>lomax:
        lorange[1]=lomax
    lazone=list(math.floor((larange[i]-lamin)/ v[0]) for i in range(2))
    lozone=list(math.floor((lorange[i]-lomin)/ v[1]) for i in range(2))#point to integer
    key=[]
    PFr(lazone, lenla,key)
    lapfr=key#integer to PFr
    key=[]
    PFr(lozone, lenlo,key)
    lopfr=key
    qhkw=qhkkw(lapfr, lopfr, 1000, 7)#PRr to locations of bloom filter
    return qhkw



"""searching on the RBLtree"""
def searchRBLtree(qhkkw,root):
 if qhkkw in root.bl:
     for ch in root.child:
         if type(ch) is int:
             result.append(root.child)
             break
         else:
             searchRBLtree(qhkkw,ch)

if __name__ == '__main__':
    """generate a RBF tree"""
    for i in range(1):#for beijing database 64, for shanghai database 4
        ttr = time.perf_counter()
        root= buildtree(df_lst[i], v_lst[i], lst[i], n, z_lst[i])
        ttre = time.perf_counter()
        print(ttre-ttr)
        v=v_lst[i]

        lenla= len(bin(math.floor((lamax-lamin)/v[0])))
        lenlo=len(bin(math.floor((lomax-lomin)/v[1])))
        step_number = [25, 50, 100, 200]
        for stn in step_number:
            print(stn)
            idx = random.randint(0, len(df_lst[i]))
            p = df_lst[i].iloc[idx]
            deta_lar = stn / 2 * v[0]
            deta_lor = stn / 2 * v[1]
            larange = []
            lorange = []
            larange.append(p['latitude'] - deta_lar)
            larange.append(p['latitude'] + deta_lar)
            lorange.append(p['longitude'] - deta_lor)
            lorange.append(p['longitude'] + deta_lor)
            # print(larange,lorange)
            tq = time.perf_counter()
            qhkw = range2hkkw(larange, lorange, lenla, lenlo)
            tqe = time.perf_counter()
            result = []
            t50 = time.time()
            searchRBLtree(qhkw, root[0])
            t50e = time.time()
            print(tqe - tq, t50e - t50)

