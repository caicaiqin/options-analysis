# -*- coding: utf-8 -*-
"""
Created on Tue May  9 16:49:54 2017

@author: Krishna Govind
"""

import pandas as pd
import numpy as np
import math as m
from scipy.stats import norm as n
import matplotlib.pyplot as plt
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D

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


#Calculating the implied Volatility using the Bisection Method
def impVol(S,K,r,t,type,MP):
    
    a = 0.0001       #Minimum Value
    b = 1        #Maximum Value
    N = 1       #Number of iterations
    tol = 10**-4

    #Anonymous function to calculate the Implied volatility based on the Bisection method
    f = lambda s:BSMOption(S,K,t,r,s,type)-MP         
    
    while (N<=2000):
        sig = (a+b)/2
        if (f(sig)==0 or (b-a)/2<tol):
            return sig
        N = N+1
        if (np.sign(f(sig))==np.sign(f(a))):
            a = sig
        else:
            b = sig

def VolSurface(S,df1,df2):
    #Moneyness
    M = df1['K']/S
    
    hT = np.median(abs(df2['T']-np.median(df2['T'])))
    hM = np.median(abs(M-np.median(M)))
    
    N = 70
    
    f = lambda z: np.exp(np.multiply(-z,z/2))/np.sqrt(2*np.pi)
    
    sT = np.linspace(min(df1['T']),max(df1['T']),N)
    sM = np.linspace(min(M),max(M),N)
    
    sIV = np.zeros([70,70])
    for i in range(0,N):
        for j in range(0,N):
            z = np.multiply(f((sT[j]-df1['T'])/hT),( f((sM[i]-M)/hM)))
            sIV[i,j] = np.sum(np.multiply(z,df1['Implied Vol']))/np.sum(z)
            
    return sM,sT,sIV


def Dupire(r,df2,S):
    
    ds = 0.1
    q = 0
    Sigma = df2['Implied Vol']
    K = df2['K']
    T = df2['T']
    MP = df2['MP']
    sig = []
    
    for i in range(0,len(K)):
        dcdt = (BSMOption(S,K[i],T[i]+ds,r,Sigma[i],'c') - BSMOption(S,K[i],T[i],r,Sigma[i],'c'))/ds
        dcdk = (BSMOption(S,K[i]+ds,T[i],r,Sigma[i],'c') - BSMOption(S,K[i],T[i],r,Sigma[i],'c'))/ds
        dcdk2 = (BSMOption(S,K[i]+ds,T[i],r,Sigma[i],'c') - 2*BSMOption(S,K[i],T[i],r,Sigma[i],'c')+BSMOption(S,K[i]-ds,T[i],r,Sigma[i],'c'))/(ds**2)
        
        lv = m.sqrt((dcdt + (r-q)*K[i]*dcdk + q*MP[i])/(0.5*(K[i]**2)*dcdk2))
        sig.append(lv)
    
    
    df = pd.DataFrame({'T':T,'K':K,'Implied Vol':Sigma,'Local Vol':sig,'MP':MP})
    df = df[['T','K','Implied Vol','Local Vol','MP']]
    #print (df)
    
    
    
    return df
    
def surface():
    
    S = 770.05
    r = 0.0066
    
    dfO = pd.read_csv("SPX.csv")
    dfO.drop_duplicates()
    
    T = dfO['T']
    K = dfO['K']
    MP = dfO['Price']
    
    Sigma = []
    BSM = []
    for i in range(0,len(T)):
        
        Sigma.append(impVol(S,K[i],r,T[i],'c',MP[i]))
        BSM.append(BSMOption(S,K[i],r,T[i],Sigma[i],'c'))
        BSM[i] = round(BSM[i],4)
    
    df1 = pd.DataFrame({'Implied Vol':Sigma,'K':K,'T':T,'MP':MP})
    df1 = df1[['K','T','Implied Vol','MP']]
    df1 = df1[df1['Implied Vol']!=0.9999389709472657]
    #print (df1)
    
    df2=df1.sort_values(by=['T','K','Implied Vol'],ascending=[True,True,True])
    
    p = list(df1['T'].unique())
    
    
    T1 = df1.loc[df1['T'] == p[2]]
    T1 = pd.DataFrame.head(T1,n=20)
    T2 = df1.loc[df1['T'] == p[3]]
    T2 = pd.DataFrame.head(T2,n=20)
    T3 = df1.loc[df1['T'] == p[4]]
    T3 = pd.DataFrame.head(T3,n=20)
    T4 = df1.loc[df1['T'] == p[5]]
    T4 = pd.DataFrame.head(T4,n=20)
    
    dfm1 = T1.merge(T2,how='outer')
    dfm2 = T3.merge(T4,how='outer')
    dfm = dfm1.merge(dfm2,how='outer')
    
    dfs=dfm.sort_values(by=['T','K','Implied Vol'],ascending=[True,True,True])
    
    sM,sT,sIV = VolSurface(S,df1,df2)
    
    fig1 = plt.figure(figsize=(9, 6))
    ax = fig1.gca(projection='3d')
    surf = ax.scatter(df2['K'], df2['T'], df2['Implied Vol'], s=20,c=None,depthshade=True)
    
    ax.set_xlabel('Strike')
    ax.set_ylabel('time-to-maturity')
    ax.set_zlabel('implied volatility')
    # Customize the z axis.
    
    
    
    
    f2 = interpolate.interp2d(sM,sT,sIV,kind='cubic')
    z = f2(sM,sT)
    
    fig2 = plt.figure(figsize=(9, 6))
    ax = fig2.gca(projection='3d')
    surf = ax.plot_surface(sT, sM, z, rstride=2, cstride=2,
                           cmap=plt.cm.coolwarm, linewidth=0.5,
                           antialiased=True)
    ax.set_xlabel('Moneyness')
    ax.set_ylabel('time-to-maturity')
    ax.set_zlabel('implied volatility')
    fig2.colorbar(surf, shrink=0.5, aspect=5)
    
    
    dz = Dupire(r,dfs,S)
    
    
    LocalVol = dz['Local Vol']
    K2 = dz['K']
    T2 = dz['T']
    MP2 = dz['MP']
    ImpliedVol = dz['Implied Vol']
    BSMLV = []
    for i in range(0,len(K2)):
        BSMLV.append(BSMOption(S,K2[i],T2[i],r,LocalVol[i],'c'))
        
    dLocal = pd.DataFrame({'K':K2,'T':T2,'Local Vol':LocalVol,'BSM':BSMLV,'MP':MP2})
    dLocal = dLocal[['K','T','Local Vol','BSM','MP']]
    print (dLocal)
    
    
    dMain = pd.DataFrame({'K':K2,'T':T2,'Local Vol':LocalVol,'Implied Vol':ImpliedVol,'BSM':BSMLV,'MP':MP2})
    dMain = dMain[['K','T','Local Vol','Implied Vol','BSM','MP']]
    print (dMain)
    
    dMain.to_csv("SPXVolatility.csv")
    
if __name__=="__main__":
    surface()