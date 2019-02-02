# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 18:28:28 2017

@author: Krishna Govind
"""

import math as m
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import norm as n
import matplotlib.pyplot as plt

#Function to calculate the Option price using BSM
def BSMOption(S,K,t,r,sigma,type):
    d1 = (m.log(S/K)+(r+(sigma**2/2))*t)/(sigma*m.sqrt(t))
    d2=d1-sigma*m.sqrt(t)
    
    if (type=='c'):    
        C = S*n.cdf(d1)-(K*m.exp(-r*t)*n.cdf(d2))
        return C
    else:    
        P = K*m.exp(-r*t)*n.cdf(-d2)-S*n.cdf(-d1)
        return P
    
#Function to calculate the Option price with the Binomial Tree
def BinomialOption(s,K,r,t,sigma,types,kind,N):
    
    S = np.zeros([N+1,N+1],dtype = float)
    O = np.zeros([N+1,N+1], dtype = float)
    deltaT = t/N
    v = r-((sigma**2)/2.0)
    deltaXU = m.sqrt(sigma**2*deltaT +(v*deltaT)**2)
    deltaXD = -deltaXU
    Pu = 0.5 + 0.5*(v*deltaT)/deltaXU

    Pd = 1-Pu
    
    if(kind=='AM' and types=='p'):
        Pu = m.exp(-r*deltaT)*Pu
        Pd = 1-Pu
    
    #Storing the values the stock prices in its respective positions based on the additive binomial tree
    for i in range(0,N+1):
        for j in range(0,i+1):
            if (i==0 and j==0):
                S[i][j] = m.exp(s)
            else:
                S[i][j] = m.exp(s + j*deltaXU + (i-j)*deltaXD)
                
    #print ("The Stock Price tree is as follows, \n",S)
    
    #Calculating the Option values at every coordinate.
    if (types =='c'):
        for z in range(0,N+1):
            O[O.shape[0]-1,z] = max(S[S.shape[0]-1,z]-K,0)
    else:
        for z in range(0,N+1):
            O[O.shape[0]-1,z] = max(K-S[S.shape[0]-1,z],0)
    
    
    for p in range(O.shape[0]-2,-1,-1):
        for q in range(0,p+1):
            if (kind == 'EU' or (kind=='AM' and types=='c')):
                O[p,q] =m.exp(-r*deltaT)*(Pu*O[p+1,q+1]+Pd*O[p+1,q])     
            else:
                O[p,q] =max(m.exp(-r*deltaT)*(Pu*O[p+1,q+1]+Pd*O[p+1,q]),K-S[p][q])
    #print ("The Option tree Prices are as xfollows, \n",O)
 
    return O[0,0]

#Calculating the implied Volatility using the Bisection Method
def impVol(S,K,r,t,type,MP):
    
    a = 0.001
    b = 1
    N = 1
    tol = 10**-4

    #Anonymous function to calculate the Implied volatility based on the Bisection method
    f = lambda s:BSMOption(S,K,t,r,s,type)-MP         
    
    while (N<=200):
        sig = (a+b)/2
        
        if (f(sig)==0 or (b-a)/2<tol):
            return sig
        
        N = N+1
        if (np.sign(f(sig))==np.sign(f(a))):
            a = sig
        else:
            b = sig

    
#Main function
def binomial():
    dfOption = pd.read_csv("GOOG.csv")
    
    S = 825.15
    r = 0.0075
    K = dfOption['Strike']
    t = dfOption['Date']
    types = dfOption['OptionType']
    bid = dfOption['Bid']
    ask = dfOption['Ask']
    BinEU = []
    BinAM = []
    BSM = []
    t2 = []
    
    for i in range(0,t.shape[0]):
        t1=datetime.strptime(t[i],'%d/%m/%Y')-datetime.strptime('15/02/2017','%d/%m/%Y')
        t2.append(t1.days/252.0)
        
        MP = (bid[i]+ask[i])/2.0
        sigma = impVol(S,K[i],r,t2[i],types[i],MP)
        BinEU.append(BinomialOption(m.log(S),K[i],r,t2[i],sigma,types[i],'EU',200))
        BinAM.append(BinomialOption(m.log(S),K[i],r,t2[i],sigma,types[i],'AM',200))
        BSM.append(BSMOption(S,K[i],t2[i],r,sigma,types[i]))
    
    df1 = pd.DataFrame({'Bin EU':BinEU,'Ask':ask,'Bid':bid,'BSM':BSM,'Bin AM':BinAM,'Option Type':types})
    
    print (df1)
        
if __name__=="__main__":
    binomial()
       
    
    
