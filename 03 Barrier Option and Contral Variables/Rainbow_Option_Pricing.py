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
import time

home = os.getcwd()
src = join(home, 'source')
out = join(home, 'output')

'''parameter'''
path = 100
steps = 1
rep = 50
dim = 2

S0_1 = 40
S0_2 = 40
K = 35
r_effective = .05
r = np.log(1+r_effective)
sig1 = 0.2
sig2 = 0.3
T = 7/12
dis_factor = np.exp(-r*T)

'''correlation matrix & cholesky'''
corr = 0.5
corrMatrix = np.array([[1,corr],[corr,1]])
upper_cho = cholesky(corrMatrix)

'''generating random numbers'''
def Rand(path, steps):
    rands = np.random.normal(0,1,(path*steps,dim))
        
    return np.inner(rands,upper_cho)

def MCS(path, steps):
    dt = T/steps
    rands = Rand(path, steps)
    
    S1_sim = np.zeros((path,steps+1))
    S2_sim = np.zeros((path,steps+1))
    
    S1_sim[:,0] = S0_1
    S2_sim[:,0] = S0_2
    
    for i in range(path):
        for j in range(1,steps+1):
            S1_sim[i,j] = S1_sim[i,j-1] * np.exp((r-pow(sig1,2)/2)*dt+sig1*np.sqrt(dt)*rands[i*steps+j-1,0])
            S2_sim[i,j] = S2_sim[i,j-1] * np.exp((r-pow(sig2,2)/2)*dt+sig2*np.sqrt(dt)*rands[i*steps+j-1,1])
            
    return S1_sim, S2_sim


Path = [100, 1000, 10000, 100000, 1000000]
result = pd.DataFrame(columns = ['a','b','c','d','CPU'])
#result = np.zeros((2*len(Path),5))
for p in Path:
    mean_a = np.zeros(rep)
    mean_b = np.zeros(rep)
    mean_c = np.zeros(rep)
    mean_d = np.zeros(rep)
    
    print('Running Path =',p,'...')
    
    t = np.zeros(rep)
    for i in range(rep):
        s = time.time()
        S1_sim, S2_sim = MCS(p, steps)
        
        # payoff
        a = np.maximum(np.maximum(S1_sim[:,-1],S2_sim[:,-1]) - K, 0) * dis_factor
        b = np.maximum(np.minimum(S1_sim[:,-1],S2_sim[:,-1]) - K, 0) * dis_factor
        c = np.maximum(K - np.maximum(S1_sim[:,-1],S2_sim[:,-1]), 0) * dis_factor
        d = np.maximum(K - np.minimum(S1_sim[:,-1],S2_sim[:,-1]), 0) * dis_factor
        
        # mean
        mean_a[i] = np.mean(a)
        mean_b[i] = np.mean(b)
        mean_c[i] = np.mean(c)
        mean_d[i] = np.mean(d)
        
        t[i] = time.time() - s
        
    result.loc[p*2,:] = [np.mean(mean_a), np.mean(mean_b), np.mean(mean_c), np.mean(mean_d), np.mean(t)]
    result.loc[p*2+1,:] = [np.std(mean_a), np.std(mean_b), np.std(mean_c), np.std(mean_d), '']
    
