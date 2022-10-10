"""安全的计算欧式距离"""
from phe import paillier # 开源库
import time # 做性能测试
import random

# print("size of private key：",paillier.DEFAULT_KEYSIZE) #512
# generate key
public_key,private_key = paillier.generate_paillier_keypair()
#
# message_list = [540.885658, 140885658]
# """encryption"""
# time_start_enc = time.time()
# encrypted_message_list = [public_key.encrypt(m) for m in message_list]
# time_end_enc = time.time()
# print("time cost (s)：",time_end_enc-time_start_enc)
#
# '''decryption'''
# time_start_dec = time.time()
# decrypted_message_list = [private_key.decrypt(c) for c in encrypted_message_list]
# time_end_dec = time.time()
# print("time cost(s)：",time_end_dec-time_start_dec)
# print("plaintext:",decrypted_message_list)

# print((private_key.decrypt(a)-private_key.decrypt(b))==private_key.decrypt(a-b))
# print(2*(private_key.decrypt(a))==private_key.decrypt(2*a))

def SSED(eml1, eml2):#eml1=[en(x1),en(x2)],eml2=[en(y1),en(y2)]
    '''cloudB'''
    r=random.randint(1,140885658)
    x1,x2=eml1
    y1,y2=eml2
    xyr1=x1-y1+public_key.encrypt(r)
    xyr2 =x2-y2 + public_key.encrypt(r)
    '''send to cloud A'''
    ar1=private_key.decrypt(xyr1)#a+r
    ar2=private_key.decrypt(xyr2)
    ear1=public_key.encrypt(ar1**2)
    ear2=public_key.encrypt(ar2**2)
    '''send to B'''
    edxy1=ear1-2*r*(x1-y1)-public_key.encrypt(r**2)
    edxy2=ear2-2*r*(x2-y2)-public_key.encrypt(r**2)
    ed2=edxy1+edxy2
    '''send to A'''
    dis2=private_key.decrypt(ed2)
    return dis2

# for i in range(10):
#     ml1=[10.2142,6.21151]
#     time_start_eloc = time.time()
#     eml1 = [public_key.encrypt(m) for m in ml1]
#     time_end_eloc = time.time()
#     ml2=[15.12543,3.215432]
#     eml2 = [public_key.encrypt(m) for m in ml2]
#     print((ml1[0]-ml2[0])**2+(ml1[1]-ml2[1])**2)
#     time_start_ssed = time.time()
#     print(SSED(eml1, eml2))
#     time_end_ssed = time.time()
#
#     print(time_end_eloc-time_start_eloc, time_end_ssed-time_start_ssed)



