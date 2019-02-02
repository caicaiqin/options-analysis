# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 16:04:34 2017

@author: Krishna Govind
"""

import numpy as np
import math as m
from scipy.stats import norm as n

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

def Binomial(s,K,t,r,sigma,types,kind,N,div,H):
    
    S = np.zeros([N+1,N+1],dtype = float)
    O = np.zeros([N+1,N+1], dtype = float)
    deltaT = t/N
    v = r-div-((sigma**2)/2.0)
    deltaXU = m.sqrt(sigma**2*deltaT +(v*deltaT)**2)
    deltaXD = -deltaXU
    
    Pu = 0.5 + 0.5*(v*deltaT)/deltaXU

    Pd = 1-Pu
    
    #Storing the values the stock prices in its respective positions based on the additive binomial tree
    for i in range(0,N+1):
        for j in range(0,i+1):
            if (i==0 and j==0):
                S[i,j] = m.exp(s)
            else:
                S[i,j] = m.exp(s + j*deltaXU + (i-j)*deltaXD)
                
    #print ("The Stock Price tree is as follows, \n",S)
    
    #Calculating the Option values at every coordinate.
    if (types =='c'):
        for z in range(0,N+1):
            
            #European Up and Out Option
            if (S[S.shape[0]-1,z]<H):    
                O[O.shape[0]-1,z] = max(S[S.shape[0]-1,z]-K,0)
            else:
                O[O.shape[0]-1,z] = 0
    else:
        for z in range(0,N+1):
            if (S[S.shape[0]-1,z]<H):
                O[O.shape[0]-1,z] = max(K-S[S.shape[0]-1,z],0)
            else:
                O[O.shape[0]-1,z] = 0
    
    for p in range(O.shape[0]-2,-1,-1):
        for q in range(0,p+1):
            if (kind == 'EU' or (kind=='AM' and types=='c')):
                #European Up and Out Option
                if (S[p,q]<H):    
                    O[p,q] =m.exp(-r*deltaT)*(Pu*O[p+1,q+1]+Pd*O[p+1,q]) 
                else:
                    O[p,q] = 0
            else:
                O[p,q] =max(m.exp(-r*deltaT)*(Pu*O[p+1,q+1]+Pd*O[p+1,q]),K-S[p][q])
                
                
    #print ("The Option tree Prices are as follows, \n",O)

    return O[0,0]    

#Function to calculate the Option price using BSM
def BSMOption(S,K,t,r,sigma,types):
    d1 = (m.log(S/K)+(r+((sigma**2)/2)*t))/(sigma*m.sqrt(t))
    d2=d1-sigma*m.sqrt(t)
    
    if (types=='c'):    
        C = S*n.cdf(d1)-(K*m.exp(-r*t)*n.cdf(d2))
        return C
    else:    
        P = K*m.exp(-r*t)*n.cdf(-d2)-S*n.cdf(-d1)
        return P
    

def dbs(S,H,t,sigma,v):
    a = (m.log(S/H) + v*t)/(sigma*(m.sqrt(t)))
    return a

def Analytical(S,K,t,r,sigma,types,kind,N,div,H):
    
    v = r-div-((sigma**2)/2.0)
    
    l = (H/S)**(2*v/(sigma**2))
    z = (H**2)/S
    
    UO1 = BSMOption(S,K,t,r,sigma,types)-BSMOption(S,H,t,r,sigma,types)
    UO2 = (H-K)*m.exp(-r*t)*n.cdf(dbs(S,H,t,sigma,v))
    UO3 = l*(BSMOption(z,K,t,r,sigma,types)-BSMOption(z,H,t,r,sigma,types)-((H-K)*m.exp(-r*t)*n.cdf(dbs(H,S,t,sigma,v))))
    
    UO= UO1-UO2-UO3
    
    print ("The European Up and Out Call Option using the analytical formula is =",UO)
    
    return UO
    
    
#TO Calculate the European Up and In call using the analytical formula and the In-Call parity
def UIC(S,K,t,r,sigma,types,kind,N,div,H,UO):
    
    v = r-div-((sigma**2)/2.0)
    
    l = (H/S)**(2*v/(sigma**2))
    z = (H**2)/S
        
    UI1 = l*(BSMOption(z,K,t,r,sigma,'p')-BSMOption(z,H,t,r,sigma,'p')+((H-K)*m.exp(-r*t)*n.cdf(-dbs(H,S,t,sigma,v))))
    UI2 = BSMOption(S,H,t,r,sigma,'c') + (H-K)*m.exp(-r*t)*n.cdf(dbs(S,H,t,sigma,v))
    
    #In Out Parity
    UI = BSMOption(S,K,t,r,sigma,types)-UO
    
    print ("The value of the European Up-and-In call option using the Analytical formula is =",UI1+UI2)
    print ("The value of the European Up-and-In call option using the In and Out Parity is =",UI)
     
    return (UI)

def AMUIP(S,K,t,r,sigma,types,kind,N,div,H):
    
    z = (H**2)/S
    
    UIP1 = (S/H)**(1-(2*(r-div))/(sigma**2))
    UIP2 = BinomialOption(m.log(z),K,r,t,sigma,'p','AM',N) - BinomialOption(m.log(z),K,r,t,sigma,'p','EU',N)
    
    EUIP = BSMOption(S,K,t,r,sigma,'p')-Binomial(m.log(S),K,t,r,sigma,'p','EU',N,div,H)
    
    UIP = UIP1*UIP2+EUIP
        
    print ("The value of the American Up and In Put option value is =",UIP)
        
def exotic():
    
    s = 10
    K = 10
    t = 0.3
    N = 200
    sigma = 0.2
    r = 0.01
    div = 0
    H = 11
    
    
    
    o=Binomial(m.log(s),K,t,r,sigma,'c','EU',N,div,H)
    print ("The European Up and Out Call Option using the Binomial tree is =",o)
       
    UO=Analytical(s,K,t,r,sigma,'c','EU',N,div,H)
   
    UIC(s,K,t,r,sigma,'c','EU',N,div,H,UO)
    
    AMUIP(s,K,t,r,sigma,'p','AM',N,div,H)

if __name__=="__main__":
    exotic()
    