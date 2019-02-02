# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:06:43 2017

@author: Krishna Govind
"""
import numpy as np
import math as m
from scipy.stats import norm as n
from time import time
import pandas as pd

np.random.seed(20000)
#Calculating the implied Volatility using the Bisection Method

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

def impVol(S,K,r,t,type,MP):
    
    a = 0.0001       #Minimum Value
    b = 1        #Maximum Value
    N = 1       #Number of iterations
    tol = 10**-4

    #Anonymous function to calculate the Implied volatility based on the Bisection method
    f = lambda s:AsianAnalytic(S,K,r,s,t)-MP         
    
    while (N<=2000):
        sig = (a+b)/2
        if (f(sig)==0 or (b-a)/2<tol):
            return sig
        N = N+1
        if (np.sign(f(sig))==np.sign(f(a))):
            a = sig
        else:
            b = sig

     
def QuestA():
    
    S0 = 959.22
    do = pd.read_csv("GOOGAsian.csv")
    K=do['K']
    T=do['T']
    r = 0.03
    
    MP = do['Price']
    M = 10000
    N = 300
    Sig = []
    priceA = []
    #a)
    for i in range(0,len(K)):
        Sig.append(impVol(S0,K[i],r,T[i],'c',MP[i]))
        priceA.append(AsianAnalytic(S0,K[i],r,Sig[i],T[i]))
    Opt = pd.DataFrame({'K':K,'T':T,'Sigma':Sig,'price':priceA,'MP':MP})
    Opt = Opt[['T','K','Sigma','price','MP']]
    print (Opt)
    
    #b)
    prices=[]
    SDs=[]
    SEs=[]
    Times=[]
    
    print ("\n")
    for i in range(0,len(K)):
        
        priceB,SDB,SEB,TimeB,Uk = AsianAMC(S0,K[i],r,Sig[i],T[i],M,N)
        prices.append(priceB)
        SDs.append(SDB)
        SEs.append(SEB)
        Times.append(TimeB)
    Opt1 = pd.DataFrame({'Price':prices,'SD':SDs,'SE':SEs,'Time':Times,'MP':MP})
    Opt1 = Opt1[['Price','SD','SE','Time','MP']]
    print (Opt1)
    #c)
    print("\n")
    prices1=[]
    SDs1=[]
    SEs1=[]
    Times1=[]
    for i in range(0,len(K)):
        priceC,SDC,SEC,TimeC,Uk2 = AsianGMC(S0,K[i],r,Sig[i],T[i],M,N)
        prices1.append(priceC)
        SDs1.append(SDC)
        SEs1.append(SEC)
        Times1.append(TimeC)
    Opt2 = pd.DataFrame({'Price':prices1,'SD':SDs1,'SE':SEs1,'Time':Times1,'MP':MP})
    Opt2 = Opt2[['Price','SD','SE','Time','MP']]
    
    print (Opt2)
    
    #d)
    print("\n")
    bnew1=[]
    Yb1=[]
    Xb1=[]
    
    for i in range(0,len(K)):
        Yb,SDAD,SEAD,TimeAD,Yi = AsianAMC(S0,K[i],r,Sig[i],T[i],M,N)
        Xb,SDBD,SEBD,TimeBD,Xi = AsianGMC(S0,K[i],r,Sig[i],T[i],M,N)
    
        b=[]
        c=[]
    
        for i in range(0,len(Xi)):
            b.append((Xi[i]-Xb)*(Yi[i]-Yb))
            c.append((Xi[i]-Xb)**2)
    
        bnew = sum(b)/sum(c)    
        bnew1.append(bnew)
        Yb1.append(Yb)
        Xb1.append(Xb)
        
    Opt3 = pd.DataFrame({'K':K,'b*':bnew1,'Yb1':Yb1,'Xb1':Xb1})
    print (Opt3)

if __name__=="__main__":
    QuestA()