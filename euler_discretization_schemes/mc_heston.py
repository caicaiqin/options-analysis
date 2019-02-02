# -*- coding: utf-8 -*-
"""
Created on Wed May  3 19:46:46 2017

@author: Krishna Govind
"""
import numpy as np
import math as m
from time import time
import pandas as pd

def EulerSchemes(S0,r,v0,kappa,theta,sigma,rho,T,K,sc):
    
    N = 100
    M = 100000
    dt = T/N
    rmse = RMSE = 0
    sum_Ct = 0
    
    time1 = time()
    for j in range(1,M+1):
        St = S0
        vt = v0
        z1 = 0
        z2 = 0
        for i in range(1,N+1):
            if (sc == 'ft'):
                vmax1 = max(vt,0)
                vt = vt + kappa*dt*(theta-vmax1) + sigma*m.sqrt(vmax1*dt)*z1
                St = St*m.exp((r-0.5*vmax1)*dt + m.sqrt(vmax1*dt)*z2)
            elif(sc=='A'):
                vmax1 = max(vt,0)
                vt = vmax1 + kappa*dt*(theta-vmax1) + sigma*m.sqrt(vmax1*dt)*z1
                St = St*m.exp((r-0.5*vmax1)*dt + m.sqrt(vmax1*dt)*z2)
            elif(sc=='R'):
                vmax1 = abs(vt)
                vt = vmax1 + kappa*dt*(theta-vmax1) + sigma*m.sqrt(vmax1*dt)*z1
                St = St*m.exp((r-0.5*vmax1)*dt + m.sqrt(vmax1*dt)*z2)  
            elif(sc=='HM'):
                vmax1 = abs(vt)
                vt = vt + kappa*dt*(theta-vt) + sigma*m.sqrt(vmax1*dt)*z1
                St = St*m.exp((r-0.5*vmax1)*dt + m.sqrt(vmax1*dt)*z2)
            else:
                vmax1 = max(vt,0)
                vt = vt + kappa*dt*(theta-vt) + sigma*m.sqrt(vmax1*dt)*z1
                St = St*m.exp((r-0.5*vmax1)*dt + m.sqrt(vmax1*dt)*z2)
            e1 = np.random.normal(0,1)
            e2 = np.random.normal(0,1)
            z1 = e1
            z2 = rho*z1 + m.sqrt(1-(rho**2))*e2
        
        Ct = max(St-K,0)
        sum_Ct = sum_Ct + Ct
        rmse = rmse +(Ct-6.8061)**2
        
    
    Timet = time()-time1    
    callt = (sum_Ct/M)*m.exp(-r*T)
    RMSE = m.sqrt(rmse)/M
    Bias = callt-6.8061
    return (callt,RMSE,Timet,Bias)

def heston():
    
    S0 = 100
    r = 0.0319
    v0 = 0.010201
    kappa = 6.21
    theta = 0.019
    sigma = 0.61
    rho = -0.7
    T = 1
    K = 100
    
    Callft,RMSEft,Timeft,Biasft = EulerSchemes(S0,r,v0,kappa,theta,sigma,rho,T,K,'ft')
    CallA,RMSEA,TimeA,BiasA = EulerSchemes(S0,r,v0,kappa,theta,sigma,rho,T,K,'A')
    CallR,RMSER,TimeR,BiasR = EulerSchemes(S0,r,v0,kappa,theta,sigma,rho,T,K,'R')
    CallHM,RMSEHM,TimeHM,BiasHM = EulerSchemes(S0,r,v0,kappa,theta,sigma,rho,T,K,'HM')
    Callpt,RMSEpt,Timept,Biaspt = EulerSchemes(S0,r,v0,kappa,theta,sigma,rho,T,K,'pt')
    
    df = pd.DataFrame(data =[[Callft,CallA,CallR,CallHM,Callpt],[RMSEft,RMSEA,RMSER,RMSEHM,RMSEpt],[Timeft,TimeA,TimeR,TimeHM,Timept],[Biasft,BiasA,BiasR,BiasHM,Biaspt]],index=['Price','RMSE','Time','Bias'])
    
    df.columns = ['Full Trunc','Absorption','Reflection','Higham and Mao','Partial Trunc']
    
    print (df)
    
if __name__=="__main__":
    heston()