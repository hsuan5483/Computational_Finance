#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Pei-Hsuan Hsu
"""

import numpy as np
import pandas as pd
import time

'''
=========
CRR Model
=========
'''
def EurCRR(n, S, K, r, sigma, t, otype):
    otype = otype.lower()
    s = time.time()
    Δt = t/n
    '''
    u = exp(sigma*sqrt(delta_t))
    '''
    u = np.exp(sigma*np.sqrt(Δt))
    d = 1./u
    
    # 風險中立機率
    p = (np.exp(r*Δt)-d) / (u-d) 
    
    # 模擬每期股價
    stockvalue = np.zeros((n+1,n+1))
    stockvalue[0,0] = S
    for i in range(1,n+1):
        stockvalue[0,i] = stockvalue[0,i-1]*u
        for j in range(1,i+1):
            stockvalue[j,i] = stockvalue[j-1,i-1]*d
    
    # option 最終節點價值
    optionvalue = np.zeros((n+1,n+1))
    for j in range(n+1):
        if otype=="c": # Call
            optionvalue[j,n] = max(0, stockvalue[j,n]-K)
        elif otype=="p": #Put
            optionvalue[j,n] = max(0, K-stockvalue[j,n])
    
    # 反推option每一節點價值
    for i in range(n-1,-1,-1):
        for j in range(i+1):
            optionvalue[j,i] = np.exp(-r*Δt)*(p*optionvalue[j,i+1]+(1-p)*optionvalue[j+1,i+1])
            
    e = time.time() - s

    return stockvalue, optionvalue, e


def AmerCRR(n, S, K, r, sigma, t, otype):  
    s = time.time()
    Δt = t/n
    '''
    u = exp(sigma*sqrt(delta_t))
    '''
    u = np.exp(sigma*np.sqrt(Δt))
    d = 1./u
    
    # 風險中立機率
    p = (np.exp(r*Δt)-d) / (u-d) 
    
    # 模擬每期股價
    stockvalue = np.zeros((n+1,n+1))
    stockvalue[0,0] = S
    for i in range(1,n+1):
        stockvalue[0,i] = stockvalue[0,i-1]*u
        for j in range(1,i+1):
            stockvalue[j,i] = stockvalue[j-1,i-1]*d
    
    # option 最終節點價值
    optionvalue = np.zeros((n+1,n+1))
    for j in range(n+1):
        if otype=="c": # Call
            optionvalue[j,n] = max(0, stockvalue[j,n]-K)
        elif otype=="p": #Put
            optionvalue[j,n] = max(0, K-stockvalue[j,n])
    
    # 反推option每一節點價值
    for i in range(n-1,-1,-1):
        for j in range(i+1):
            # max(提前履約價值,繼續持有價值)
            if otype=="p":
                optionvalue[j,i] = max(K-stockvalue[j,i], np.exp(-r*Δt)*(p*optionvalue[j,i+1]+(1-p)*optionvalue[j+1,i+1]))
            elif otype=="c":
                optionvalue[j,i] = max(stockvalue[j,i]-K, np.exp(-r*Δt)*(p*optionvalue[j,i+1]+(1-p)*optionvalue[j+1,i+1]))
    
    e = time.time() - s
    
    return optionvalue[0,0], e    
    
