#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Pei-Hsuan Hsu
"""

import os
from os.path import join
import numpy as np
import pandas as pd
from numpy.linalg import cholesky
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import time
np.random.seed(12345)

def LSM_1(S0, K, sigma, r, T, steps, paths, degree=2, optype='c'):
    '''
    =================   ========
        parameter         type
    =================   ========
    S0                  float
    K                   float
    sigma               float
    r                   float
    T                   float
    steps               int
    paths               int
    degree              int
    optype              string
    '''
    
    dt = T/steps
    dis_factor = np.exp(-r*dt)
    
    # Simulation of stock price
    sim_S = np.zeros((paths, steps+1))
    sim_S[:, 0] = S0
    for step in range(steps):
        # Generate Random Numbers
        dz = np.random.normal(0,1,(paths))
        sim_S[:, step+1] = sim_S[:, step] * np.exp((r - np.power(sigma, 2) / 2)*dt + sigma*np.sqrt(dt)*dz)

    # 儲存每期價值 payoff
    V = np.zeros((paths, steps+1))
    if optype == 'c':
        payoff = lambda X: np.maximum(X - K, 0)
    else:
        payoff = lambda X: np.maximum(K - X, 0)

    V[:, -1] = payoff(sim_S[:, -1])
    
    # 儲存已結束的path index
    I = []
    
    # LSM
    '''
    step = steps-3
    '''
    for step in reversed(range(1,steps)):
        # 立刻執行價值
        V[:, step] = payoff(sim_S[:, step])
        ### 結束的path歸零
        V[I, step] = 0
        index = np.where(V[:, step] > 0)[0]
        if len(index) == 0:
            break
        
        # regression
        model = make_pipeline(PolynomialFeatures(degree, interaction_only=True), LinearRegression())
        X = sim_S[index, step].reshape(len(index),1)
        y = (V[index, step+1] * dis_factor).reshape(len(index),1)
        model.fit(X, y)
        
        # 繼續持有價值
        EX = model.predict(X)
                
        for i, p in enumerate(index):
            V[p, step] = np.where(V[p, step] > EX[i], V[p, step], 0)
            
        index2 = np.where(V[:, step] > 0)[0]
        V[index2, step+1] = 0
        
        I = np.hstack((I, np.where(V[:, step+1] > 0)[0]))
        I = np.unique(I).astype(int)
        
    # pricing
    P = 0
    VV = np.sum(V, axis=0)
    for t, v in enumerate(VV):
        P += v*np.exp(-r*(t*dt))/paths
    
    return P

# parameters
rep = 1
degree = 2
steps = 50
paths = 1000

T = 1
S0 = 40
K = 50
sigma = 0.6
r=0.05
optype='p'

p = []
s = time.time()
for i in range(10):
    p.append(LSM_1(S0, K, sigma, r, T, steps, paths, degree, optype))
e = time.time() - s

print('LSM_1')
print('put =', np.mean(p))
print('put(std) =', np.std(p))
print('time =', e/len(p))


import Black_Scholes as bs
bs_p = bs.black_scholes('p', S0, K, T, r, sigma)
print('black sholes =', bs_p)
   