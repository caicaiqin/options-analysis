# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 02:26:07 2017

@author: Krishna Govind
"""
import numpy as np

import math as m


def CF(xt,vt,tau,mu,a,uj,bj,rho,sig,phi,i):
    
    
    xj = bj-rho*sig*np.dot(phi,i)
    
    dj = np.sqrt(xj**2-(sig**2)*(2*uj*phi*complex(0,1)-phi**2))
    
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

    return (C )
    

def Heston():
        
    S = 1
    K = 1
    q = 0
    r = 0
    k = 1
    sigma = 0.2
    rho = -0.3
    theta = 0.1
    v0 = 0.1
    lda = 0
    T = 5
    
    i = complex(0,1)

    C = HestonCall(S,K,r,T,v0,k,theta,sigma,rho,lda,i,q)
    
    print ("The price of the Call Option using Heston Model is: ",C)


if __name__=="__main__":
    Heston()