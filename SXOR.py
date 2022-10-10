import random
import time # 做性能测试
# Find the greatest common divisor
def gcd(a, b):
    if a < b:
        return gcd(b, a)
    elif a % b == 0:
        return b
    else:
        return gcd(b, a % b)


# Quick power & modulo
def power(a, b, c):
    ans = 1
    while b != 0:
        if b & 1:
            ans = (ans * a) % c
        b >>= 1
        a = (a * a) % c
    return ans


# Lucas-lemmer sex test
def Lucas_Lehmer(num) -> bool:  # Quickly check whether POw (2,m)-1 is prime
    if num == 2:
        return True
    if num % 2 == 0:
        return False
    s = 4
    Mersenne = pow(2, num) - 1  # Pow (2, num)-1 is a Mersenne number
    for x in range(1, (num - 2) + 1):  # Num-2 is the number of cycles, and +1 means the right interval is open
        s = ((s * s) - 2) % Mersenne
    if s == 0:
        return True
    else:
        return False


# Detection of large prime numbers
def Miller_Rabin(n):
    a = random.randint(2,n-2) # Randomly find 'a' which is belong to [2,n-2]
    s = 0                     # S is the power of factor 2 in d
    d = n - 1
    while (d & 1) == 0:       # Let's factor out all the 2's in d.
        s += 1
        d >>= 1

    x = power(a, d, n)
    for i in range(s):        # Perform s secondary probes
        newX = power(x, 2, n)
        if newX == 1 and x != 1 and x != n - 1:
            return False      # Using the inverse of the quadratic theorem, n is determined to be composite.
        x = newX

    if x != 1:                # If x=a^(n-1) (mod n), then n is determined to be composite.
        return False

    return True               # Judge by the converse of Fermat's little theorem. The number that survives this test
                              # is most likely prime.


# Extended Euclidean algorithm, ab=1 (mod m), yields the multiplicative inverse b of A under module m
def Extended_Eulid(a: int, m: int) -> int:
    def extended_eulid(a: int, m: int):
        if a == 0:  # boundary condition
            return 1, 0, m
        else:
            x, y, gcd = extended_eulid(m % a, a)  # recursion
            x, y = y, (x - (m // a) * y)  # Recursively, the left end is the upper layer
            return x, y, gcd              # Returns the result of the first level calculation.
        # The final return y value is the multiplication inverse of b in mode a
        # If y is complex, y plus a is the corresponding positive inverse

    n = extended_eulid(a, m)
    if n[1] < 0:
        return n[1] + m
    else:
        return n[1]


# Generate the field parameter p, approximately 512bits in length
def Generate_p() -> int:
    a = random.randint(10**150, 10**160)
    while gcd(a, 2) != 1:
        a = random.randint(10**150, 10**160)
    return a


# Generate the field parameter alpha
def Generate_alpha(p: int) -> int:
    return random.randint(2, p)


# Generate a prime number less than p, approximately 512bits long, as the private key
def Generate_private_key(p: int) -> int:
    pri = random.randint(2, p - 2)
    while gcd(pri, p-1) != 1:
        pri = random.randint(2, p - 2)
    return pri


# Bob or Alice uses the field parameter P and the resulting mask to encrypt the message
def encrypt(message, p, r, Public_key) -> int:
    T2 = power(Public_key, r, p)
    gr = power(alpha, r, p)
    T1 = message ^hash(gr)
    return T1,T2


# Bob or Alice decrypts the ciphertext using the field parameter P and the masks obtained by them
def decrypt(T1, T2, inverse_element, p):
    plaintext = T1^hash(power(T2, inverse_element, p))
    return plaintext


def quick_power(a: int, b: int) -> int:
    ans = 1
    while b != 0:
        if b & 1:
            ans = ans * a
        b >>= 1
        a = a * a
    return ans


def Generate_prime(key_size: int) -> int:
    while True:
        num = random.randrange(quick_power(2, key_size - 1), quick_power(2, key_size))
        num2 = (num - 1) % 2
        if num2==0:
            if Miller_Rabin(num):
            # if Miller_Rabin(num) and Miller_Rabin(int((num-1)/2)):
            #      print((num-1)/2)
                 return num, int((num-1)/2)
        else:
            continue




p,q = Generate_prime(1024)
alpha = Generate_alpha(p)
message=20220128
Private_key = Generate_private_key(p)
Public_key = power(alpha, Private_key, p)
r = Generate_private_key(p)
# te1= time.time()
# T1, T2 = encrypt(message, p, r, Public_key)
# te2= time.time()
# inverse_element = Extended_Eulid(Private_key, p-1)
# plaintext = decrypt(T1, T2, inverse_element, p)
# td2=time.time()
# print('encryption',te2-te1)
# print('decryption',td2-te2)
# print("Domain parameters: ")
# print("p:            ", p)
# print("q:            ", q)
# print("g:        ", alpha)
# print("Plaintext:   ", plaintext)

def SXOR(T1,T2,t1,t2,inverse_element, p):
    M=T1^t1
    N=M^hash(power(T2, inverse_element, p))^hash(power(t2, inverse_element, p))
    if N==0:
        return True
    else:
        return False

# for i in range(10):
#     message1=20220228
#     T1, T2 = encrypt(message, p, r, Public_key)
#     r1 = Generate_private_key(p)
#     t1, t2 = encrypt(message1, p, r1, Public_key)
#     time_start_ssor = time.time()
#     SXOR(T1,T2,t1,t2,inverse_element, p)
#     time_end_ssor = time.time()
#     print(time_end_ssor-time_start_ssor)
#     #时间0.0110