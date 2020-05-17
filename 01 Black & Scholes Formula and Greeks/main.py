#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Pei-Hsuan Hsu
"""

import os
from os.path import join
import pandas as pd
import numpy as np
import Black_Scholes as bs
import Monte_Carlo_Simulation as mcs
import time

path = os.getcwd()
out = join(path , 'output')

# parameters
S = 50
X = 60
r = 0.05
sigma = 0.3
T = 0.5

'''
===================
Black-Scholes Model
===================
'''
def BS(S, X, T, r, sigma):
    
    result = {}
    s = time.time()
    result['Call'] = [bs.black_scholes('c', S, X, T, r, sigma, True)]
    result['Put'] = [bs.black_scholes('p', S, X, T, r, sigma, True)]
    result['Delta_Call'] = [bs.delta('c', S, X, T, r, sigma)]
    result['Delta_Put'] = [bs.delta('p', S, X, T, r, sigma)]
    result['Gamma'] = [bs.gamma('c', S, X, T, r, sigma)]
    result['Vega'] = [bs.vega('c', S, X, T, r, sigma)]
    result['Rho_Call'] = [bs.rho('c', S, X, T, r, sigma)]
    result['Rho_Put'] = [bs.rho('p', S, X, T, r, sigma)]
    result['Theta_Call'] = [bs.theta('c', S, X, T, r, sigma)]
    result['Theta_Put'] = [bs.theta('p', S, X, T, r, sigma)]
    result['CPU time'] = [time.time()-s]
    
    return result

'''
======================
Monte Carlo Simulation
======================
'''
d = [0.01, 0.001]
paths = [100, 1000, 10000, 100000]#, 1000000]
steps = 250

index = ['Black-Scholes']
index.extend(['MCS('+str(x)+')' for x in paths])

for delta in d:
    
    filename = 'results(d='+str(delta)+').xlsx'
    result = BS(S, X, T, r, sigma)
    
    for path in paths:
        print('Running d = '+str(delta)+' path = '+str(path)+'......')
                
        call, put, calldelta, putdelta, gamma, vega, callrho, putrho, calltheta, puttheta, t = \
        mcs.MCS_Options(S, X, r, T, sigma, path, steps, True, True, join(out,'Simulation(d='+str(delta)+',path num='+str(path)+').png'))
        
        result['Call'].append(call)
        result['Put'].append(put)
        result['Delta_Call'].append(calldelta)
        result['Delta_Put'].append(putdelta)
        result['Gamma'].append(gamma)
        result['Vega'].append(vega)
        result['Rho_Call'].append(callrho)
        result['Rho_Put'].append(putrho)
        result['Theta_Call'].append(calltheta)
        result['Theta_Put'].append(puttheta)
        result['CPU time'].append(t)
        
    result = pd.DataFrame(result, index=index)
    result.to_excel(join(out,filename), index = True)

