# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 13:36:39 2017

@author: Krishna Govind
"""

import pandas as pd
import math as m
import numpy as np
from datetime import datetime
from scipy.stats import norm as n
import matplotlib.pyplot as plt


#Function to calculate the Option price using BSM
def BSMOption(S,K,t,r,sigma,type):
    
    d1 = (np.log(S/K)+(r+(sigma**2/2))*t)/(sigma*np.sqrt(t))
    d2=d1-sigma*np.sqrt(t)
    
    if (type=='c'):    
        C = S*n.cdf(d1)-(K*np.exp(-r*t)*n.cdf(d2))
        return C
    else:    
        P = K*np.exp(-r*t)*n.cdf(-d2)-S*n.cdf(-d1)
        return P


#Calculating the implied Volatility using the Bisection Method
def bisect(S,K,r,t,types,MP):
    
    a = 0.0001       #Minimum Value
    b = 0.9999        #Maximum Value
    N = 1       #Number of iterations
    tol = 10**-4

    #Anonymous function to calculate the Implied volatility based on the Bisection method
    f = lambda s:BSMOption(S,K,t,r,s,types)-MP         
    
    while (N<=200):
        sig = (a+b)/2
        if (f(sig)==0 or (b-a)/2<tol):
            return sig
        N = N+1
        if (np.sign(f(sig))==np.sign(f(a))):
            a = sig
        else:
            b = sig
            
#Calculating the implied Volatility using the Secant Method
def secant(S,K,r,t,types,MP):
    
    x0 = 0.09
    xx = 0.01
    tolerance = 10**(-7)
    epsilon = 10**(-14)
    
    maxIterations = 200
    SolutionFound = False
    
    #Anonymous function to calculate the Implied volatility using the Secant Method
    f = lambda s:BSMOption(S,K,t,r,s,types)-MP         
    
    for i in range(1,maxIterations+1):
        y = f(x0)
        yprime = (f(x0)-f(xx))/(x0-xx)      
        
        if (abs(yprime)<epsilon):
            break
        
        x1 = x0 - y/yprime
        
        if (abs(x1-x0)<=tolerance*abs(x1)):
            SolutionFound = True
            break
        
        x0 = x1
    
    if (SolutionFound):
        return x1
    else:
        print ("Did not converge") 
        
#Calculating the implied Volatility using the Newton Method
def newton(S,K,r,t,types,MP):
    
    x0 = 1
    xx = 0.001
    tolerance = 10**(-7)
    epsilon = 10**(-14)
    
    maxIterations = 200
    SolutionFound = False
    
    #Anonymous function to calculate the Implied volatility using the Newton Method
    f = lambda s:BSMOption(S,K,t,r,s,types)-MP         
    
    for i in range(1,maxIterations+1):
        y = f(x0)
        yprime = (f(x0+xx)-f(x0-xx))/(2*x0*xx)      #Using Central Difference method to find the derivative
        
        if (abs(yprime)<epsilon):
            break
        
        x1 = x0 - y/yprime
        
        if (abs(x1-x0)<=tolerance*abs(x1)):
            SolutionFound = True
            break
        
        x0 = x1
    
    if (SolutionFound):
        return x1
    else:
        print ("Did not converge")

def VolAprox():
    #Stock price at the time of analysis
    S = 862.76      #Stock Price of Google
    r = 0.0075      #Risk Free Rate      
    
    df = pd.read_csv("GOOG.csv")        #Importing the data from the CSV file
    df = df.dropna()
    
    T = df['Date']
    Bid = df['Bid']
    Ask = df['Ask']
    K = df['Strike']
    OptType = df['OptionType']
    CM = (Bid+Ask)/2.0      #Market Value
    
    t =[]
    volb = [];vols=[];voln=[]
    
    
    #Loop over options in the CSV file
    for i in range(0,len(OptType)):
        t1=datetime.strptime(T[i],'%d/%m/%Y')-datetime.strptime('25/04/2017','%d/%m/%Y')    
        t.append(t1.days/252.0)
        volb.append(bisect(S,K[i],r,t[i],OptType[i],CM[i]))
        vols.append(secant(S,K[i],r,t[i],OptType[i],CM[i]))
        voln.append(newton(S,K[i],r,t[i],OptType[i],CM[i]))
        
    #Showcasing the values
    df1 = pd.DataFrame({'Bisection Method':volb,'Secant Method':vols,'Newton Method':vols,'Market Price':CM})
    print (df1)
    
    
if __name__=="__main__":
    VolAprox()