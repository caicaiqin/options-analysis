# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 02:26:07 2017

@author: Krishna Govind
"""
import numpy as np
import math as m
import pandas as pd
from datetime import datetime
from scipy.optimize import least_squares
import sys

S = 905.96
df = pd.read_csv("GOOG.csv")
K = df['Strike']
T = df['Date']
CM = (df['Ask']+df['Bid'])/2.0
r = 0
q = 0
i = complex(0,1)
t = []
for j in range(0,len(K)):
    t1=datetime.strptime(T[j],'%d/%m/%Y')-datetime.strptime('29/04/2017','%d/%m/%Y')    
    t.append((t1.days/252.0))

def CF(xt,vt,tau,mu,a,uj,bj,rho,sig,phi,i):
        
    xj = bj-rho*sig*np.dot(phi,i)
    
    dj = np.sqrt(xj**2-(sig**2)*(2*uj*phi*i-phi**2))
    
    gj = np.divide(xj+dj,xj-dj)
    
    D = ((xj+dj)/(sig**2))*(1-np.exp(dj*tau))/(1-gj*np.exp(dj*tau))
    
    xx = (1-gj*np.exp(dj*tau))/(1-gj)
    
    C = mu*phi*i*tau+(a/(sig**2))*((xj+dj)*tau-2*np.log(xx))
    
    fj = np.exp(C+D*vt+i*phi*xt)
    
    return fj
    
def HestonCall(S,K,r,T,v0,k,theta,sigma,rho,lda,i,q):
    
    dphi = 0.01
    maxphi = 50
    eps = 2.2204e-16
    phi = np.arange(eps,maxphi,dphi)
    
      
    f1 = CF(m.log(S),v0,T,0,k*theta,0.5,k+lda-rho*sigma,rho,sigma,phi,i)
    
    P1 = 0.5+(1/m.pi)*sum(np.real(np.exp(-i*phi*np.log(K))*f1/(i*phi)*dphi))
    
    f2 = CF(m.log(S),v0,T,0,k*theta,-0.5,k+lda,rho,sigma,phi,i)
    
    P2 = 0.5+(1/m.pi)*sum(np.real(np.exp(-i*phi*np.log(K))*f2/(i*phi)*dphi))
    
    C = S*P1-K*np.exp(-(r-q)*T)*P2

    return (C)

#minimizing this function
def f(x0,S,K,t,i,q,CM):
    
    f1= CM-HestonCall(S,K,r,t,x0[0],x0[1],x0[2],x0[3],x0[4],0,i,q)
    return f1

def Heston():
    
    x0 = np.array([0.5,0.5,0.5,0.05,0.5])
    lb = [0, 0, 0, 0, -.9]
    ub = [1, 20, 1, .5, .9]
    x1=[]
    
    #Minimizing the square error using the scipy's least_squares function similar to the one in matlab
    if (2*x0[1]*x0[2]>x0[3]**2):
        for p in range(0,len(K)):
            x1.append(least_squares(f,x0,bounds=(lb,ub),args=(S,K[p],t[p],i,q,CM[p])).x)
    else:
        print ("Choose Better Initial estimates before you continue")
        sys.exit(0)
    
    v=[]
    k=[]
    theta = []
    vv = []
    rho = []
    price=[]
    for z in range(0,len(K)):
        v.append(x1[z][0])
        k.append(x1[z][1])
        theta.append(x1[z][2])
        vv.append(x1[z][3])
        rho.append(x1[z][4])
        price.append(HestonCall(S,K[z],r,t[z],v[z],k[z],theta[z],vv[z],rho[z],0,i,q))
    
    
    params = pd.DataFrame({'Vt':v,'Kappa':k,'Theta':theta,'VolVol':vv,'Rho':rho,'Heston Option Price':price,'Date':T})
    params = params[['Date','Vt','Kappa','Theta','VolVol','Rho','Heston Option Price']]
    
    print (params)
    
    #Using the Penalty factor to get better results
    v1 = abs((v-x0[0])**2)+v
    k1 = abs((k-x0[1])**2)+k
    theta1 = abs((theta-x0[2])**2)+theta
    vv1 = abs((vv-x0[3])**2)+vv
    rho1 = abs((rho-x0[4])**2)+rho
    
    x3=[]
    for p in range(0,len(K)):
        v2 = v1[0]
        k2 = k1[0]
        theta2 = theta1[0]
        vv2 = vv1[0]
        rho2 = rho1[0]
        xp = [v2,k2,theta2,vv2,rho2]
        x3.append(least_squares(f,xp,bounds=(lb,ub),args=(S,K[p],t[p],i,q,CM[p])).x)
    
    v3=[]
    k3=[]
    theta3 = []
    vv3 = []
    rho3 = []
    price3=[]
    for z in range(0,len(K)):
        v3.append(x3[z][0])
        k3.append(x3[z][1])
        theta3.append(x3[z][2])
        vv3.append(x3[z][3])
        rho3.append(x3[z][4])
        price3.append(HestonCall(S,K[z],r,t[z],v3[z],k3[z],theta3[z],vv3[z],rho3[z],0,i,q))
    
    
    params3 = pd.DataFrame({'Vt':v3,'Kappa':k3,'Theta':theta3,'VolVol':vv3,'Rho':rho3,'Heston Option Price':price3,'Date':T})
    params3 = params3[['Date','Vt','Kappa','Theta','VolVol','Rho','Heston Option Price']]
    
    print ("\n--------------\n")
    print (params3)
    
if __name__=="__main__":
    Heston()
