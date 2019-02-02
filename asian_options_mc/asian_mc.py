# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:06:43 2017

@author: Krishna Govind
"""
import numpy as np
import math as m
from scipy.stats import norm as n
from time import time

np.random.seed(20000)
def AsianAnalytic(S0,K,r,sigma,T):
    
    N = 252*T
    
    adj = sigma*m.sqrt((2*N +1)/(6*(N+1)))
    
    p = 0.5*(r-0.5*sigma**2 + adj**2)
    
    d1 = (1/(m.sqrt(T)*adj))*(m.log(S0/K)+(p+0.5*adj**2)*T)
    
    d2 = (1/(m.sqrt(T)*adj))*(m.log(S0/K)+(p-0.5*adj**2)*T)
    
    price = m.exp(-r*T)*(S0*m.exp(p*T)*n.cdf(d1) - K*n.cdf(d2))
    
    return price

def AsianAMC(S0,K,r,sigma,T,M,N):
    dt = T/N
    nudt = (r-0.5*sigma**2)*dt
    sigsdt = sigma*m.sqrt(dt)
    
    sum_CT = 0
    sum_CT2 = 0
    
    time1 = time()
    C = []
    for j in range(1,M+1):
        
        St = S0
        sum_s = 0
        
        for i in range(1,N+1):
            e = np.random.normal(0,1)
            St = St*m.exp(nudt - sigsdt*e)
            sum_s = sum_s + St

        A = sum_s/N
        CT = max(A-K,0)
        sum_CT = sum_CT + CT
        sum_CT2 = sum_CT2 +CT*CT
        C.append(CT)
        
    
    TimeTaken = time()-time1
    call = (sum_CT/M)*m.exp(-r*T)
    SD = m.sqrt((sum_CT2-sum_CT*sum_CT/M)*m.exp(-2*r*T)/(M-1))
    SE = SD/m.sqrt(M)
    
    return (call,SD,SE,TimeTaken,C)

def AsianGMC(S0,K,r,sigma,T,M,N):
    dt = T/N
    nudt = (r-0.5*sigma**2)*dt
    sigsdt = sigma*m.sqrt(dt)
    
    sum_CT = 0
    sum_CT2 = 0
    
    time1 = time()
    C = []
    for j in range(1,M+1):
        
        St = S0
        prod_s = 1
        
        for i in range(1,N+1):
            e = np.random.normal(0,1)
            St = St*m.exp(nudt - sigsdt*e)
            prod_s = prod_s * (St**(1/N))

        G = prod_s
        CT = max(G-K,0)
        sum_CT = sum_CT + CT
        sum_CT2 = sum_CT2 +CT*CT
        C.append(CT)

    TimeTaken = time()-time1
    call = (sum_CT/M)*m.exp(-r*T)
    SD = m.sqrt((sum_CT2-sum_CT*sum_CT/M)*m.exp(-2*r*T)/(M-1))
    SE = SD/m.sqrt(M)
    
    return (call,SD,SE,TimeTaken,C)
        
def QuestA():
    
    S0 = 100
    K = 100
    r = 0.03
    sigma = 0.3
    T = 5
    M = 10000
    N = 300
    #a)
    priceA = AsianAnalytic(S0,K,r,sigma,T)
    print ("\nThe price of a geometric Asian Option in the Black Scholes Model is : ",priceA)
    
    #b)
    print ("\n")
    priceB,SDB,SEB,TimeB,Uk = AsianAMC(S0,K,r,sigma,T,M,N)
    cdB1 = priceB+1.96*SEB
    cdB2 = priceB-1.96*SEB
    print ("The price of the Asian Arithmetic Option using MC is :",priceB)
    print ("The Standard Error is :",SEB)
    print ("The Time taken for the simulation is :",TimeB)
    print ("The confidence interval is between",cdB2,"-",cdB1)
    
    #c)
    print("\n")
    priceC,SDC,SEC,TimeC,Uk2 = AsianGMC(S0,K,r,sigma,T,M,N)
    cdC1 = priceC+1.96*SEC
    cdC2 = priceC-1.96*SEC
    print ("The price of the Asian Geometric Option using MC is :",priceC)
    print ("The Standard Error is :",SEC)
    print ("The Time taken for the simulation is :",TimeC)
    print ("The confidence interval is between",cdC2,"-",cdC1)
    
    #d)
    print("\n")
    Yb,SDAD,SEAD,TimeAD,Yi = AsianAMC(S0,K,r,sigma,T,M,N)
    Xb,SDBD,SEBD,TimeBD,Xi = AsianGMC(S0,K,r,sigma,T,M,N)
    
    b=[]
    c=[]
    
    for i in range(0,len(Xi)):
        b.append((Xi[i]-Xb)*(Yi[i]-Yb))
        c.append((Xi[i]-Xb)**2)
    
    bnew = sum(b)/sum(c)    
    print ("The value of b* is :",bnew)
    print ("The Arithmetic Option is :",Yb)
    print ("The Geometric Option is :",Xb)
        
    #e)
    Error = priceA - priceC
    print("\n")
    print ("The Error is :",Error)
    
    
    #f)
    Pa = Yb - (bnew)*Error
    print("\n")
    print ("The modified Arithmetic Option Price is :",Pa)

if __name__=="__main__":
    QuestA()