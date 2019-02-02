# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 21:01:37 2017

@author: Krishna Govind
"""
import math as m
import numpy as np
from scipy.stats import norm as n
import pandas as pd
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


def Trinomial(s,K,t,r,sigma,types,kind,div,N):
    
    S=np.zeros(((2*N)+1,(2*N)+1))
    O=np.zeros(((2*N)+1,(2*N)+1))
   
    dt=t/N
    v = r-div-((sigma**2)/2.0)
    dx = 0.103
            
    deltaXU = m.exp(dx)
    deltaXD = m.exp(-dx)
       
    pu = 0.5*(((sigma**2)*dt+(v*dt)**2)/(dx**2)+(v*dt)/dx)
    pm = 1-((sigma**2)*dt+((v*dt)**2))/(dx**2)
    pd = 1-(pm+pu)
    
    #Storing the Stock prices into the Numpy array
    k = int((2*N+1)/2)
    S[k,0] = m.exp(s) 
    for j in range(1,N+1):
        for i in range(k-j+1,k+j-1+1):    
            S[i-1,j] = S[i,j-1]*deltaXU
            S[i,j] = S[i,j-1]
            S[i+1,j] = S[i,j-1]*deltaXD
            
    
    #Calculating the Option prices of the extreme nodes
    if (types =='c'):
        for z in range(0,2*N+1):
            O[z,N] = max(S[z,N]-K,0)
            
    else:
        for z in range(0,2*N+1):
            O[z,N] = max(K-S[z,N],0)
    
    #Calculating the option price        
    for j in range(N,0,-1):
        for i in range(N+1-j+1,N+1+j-1+1):
            if (kind == 'EU' or (kind=='AM' and types=='c')):
                O[i-1,j-1] = m.exp(-r*dt)*(pu*O[i-2,j]+pm*O[i-1,j]+pd*O[i,j])
            else:
                O[i-1,j-1] = max(m.exp(-r*dt)*(pu*O[i-2,j]+pm*O[i-1,j]+pd*O[i,j]),K-S[i-1,j-1])
    
    
    
    return O[k,0]
 

#Main function which calls all the functions
def trinomial():
    S = 100
    K = 100
    t = 8/12.0 
    sigma = 0.03
    r = 0
    div = 0
    N = 2
    
    BSMC = BSMOption(S,K,t,r,sigma,'c')
    BSMP = BSMOption(S,K,t,r,sigma,'p')
    tri1=Trinomial(m.log(S),K,t,r,sigma,'p','EU',div,N)
    tri2=Trinomial(m.log(S),K,t,r,sigma,'p','AM',div,N)
    tri3=Trinomial(m.log(S),K,t,r,sigma,'c','EU',div,N)
    tri4=Trinomial(m.log(S),K,t,r,sigma,'c','AM',div,N)
    print ("\nThe Value of the European Put Option using the Trinomial Tree is =",tri1)
    print ("  The Value of the American Put Option using the Trinomial Tree is =",tri2)
    print ("  The Value of the European Call Option using the Trinomial Tree is =",tri3)
    print ("  The Value of the American Call Option using the Trinomial Tree is =",tri4)
    
    print ("\nThe Black Scholes Value of the European Put =",BSMP)
    print ("\nThe Black Scholes Value of the European Call =",BSMC)
    
    index = ['Trinomial Eu Put','Trinomial AM Put','Trinomial EU Call','Trinomial AM Call','BSM Call','BSM Put']
    df = pd.DataFrame([tri1,tri2,tri3,tri4,BSMC,BSMP], index = index)
    print ("\nThe following is the table of the values of European and American versions of Options, \n",df)
    
   
    
    N1 = [1,2,3,4,5,6,7,8,9,10]
    trip1=[]
    trip2=[]
    trip3=[]
    trip4=[]
    for z in range(0,len(N1)):
        trip1.append(Trinomial(m.log(S),K,t,r,sigma,'p','EU',div,N1[z]))
        trip2.append(Trinomial(m.log(S),K,t,r,sigma,'p','AM',div,N1[z]))
        trip3.append(Trinomial(m.log(S),K,t,r,sigma,'c','EU',div,N1[z]))
        trip4.append(Trinomial(m.log(S),K,t,r,sigma,'c','AM',div,N1[z]))

    
    df1 = pd.DataFrame({'Tri EU Put':trip1,'BSM Put':BSMP,'Tri AM Put':trip2,'Tri EU Call':trip3,'Tri AM Call':trip4},index=N1)
    
    print (df1)
    
if __name__=="__main__":
    
    trinomial()