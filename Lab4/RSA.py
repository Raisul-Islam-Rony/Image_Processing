# -*- coding: utf-8 -*-
"""
Created on Thu May 11 10:14:02 2023

@author: Raisul
"""

import math

def gcd(a,h):
    temp=0;
    while(1):
        temp= a % h 
        if(temp==0):
            return h
        a=h
        h=temp
def mod_inverse(phi,e):
    t1=0
    t2=1
    while(1):
        if(e==0):
            break
        q=phi//e
        r=phi%e
        t=t1 - t2*q
        phi=e
        e=r
        t1=t2
        t2=t
    return t1

def big_mod(a,b,c):
   if(b==0):
       return 1
   if(b%2==0):
       p=big_mod(a,b//2,c)
       p=p%c
       return (p*p)%c
   else:
       p=a%c
       q=big_mod(a, b-1, c)
       q=q%c
       return (p*q)%c

        

def power_modulo(a, b, n):
    result = 1
    while b > 0:
        if b % 2 == 1:
            result = (result * a) % n
        a = (a * a) % n
        b //= 2
    return result   




p=957529
q=402881
n=p*q;
phi=(p-1)*(q-1)

e=402881
print("Value of Phi ",phi)
print(mod_inverse(18, 5))

print(big_mod(2,3,4))

while(e<phi):
    if(gcd(phi,e)==1):
        break 
    else:
        e=e+1
k = 0

while((1+k*phi)%e!=0):
    k+=1

d=mod_inverse(phi, e)

ms=12

print("Public Key : ",e)        
print("Private key :",d)
print("Actual Message : ",ms)

c=pow(ms,e,n)
print(big_mod(ms, e, n))
print(power_modulo(c,d,n))
print("Encrypted Message ",c)

m=pow(c,d,n);

print("Dycrypted Message : ",m)