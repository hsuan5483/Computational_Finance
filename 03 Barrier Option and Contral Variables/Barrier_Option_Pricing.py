#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Topic: Pricing Barrier Option by MCS with Variance Reduction Techniques

@author: Pei-Hsuan Hsu

reference:
# contral variate
https://www.goddardconsulting.ca/matlab-monte-carlo-assetpaths-contvars.html
"""

import os
from os.path import join
import numpy as np
import pandas as pd
import Black_Scholes as bs
import time

np.random.seed(12345)

home = os.getcwd()
src = join(home, 'source')
out = join(home, 'output')

'''parameter'''
path = 100
steps = 250
rep = 50

S0 = 50
H = 40
K = 50
r = 0.05
sig = 0.3
T = 1
d = 0.01
dis_factor = np.exp(-r*T)

result = pd.DataFrame()

bs_call = bs.black_scholes('c', S0, K, T, r, sig)
bs_put = bs.black_scholes('p', S0, K, T, r, sig)
# barrier option price > monte carlo option price
# 同一個方法寫在一起比較好

'''
=================
Crude Monte Carlo
=================
'''
def MCS(optype, S, K, T, r, sig, path, steps):
    dt = T/steps
    rands = np.random.normal(0,1,(path*steps,1))
    
    sim = np.zeros((path,steps+1))
    
    sim[:,0] = S
    
    for i in range(path):
        for j in range(1,steps+1):
            sim[i,j] = sim[i,j-1] * np.exp((r-pow(sig,2)/2)*dt+sig*np.sqrt(dt)*rands[i*steps+j-1,0])
            if sim[i,j] < H:
                break
            
    price = np.zeros(sim.shape[0])
    for i in range(sim.shape[0]):
        if sim[i,-1] == 0:
            price[i] = 0
        elif optype.lower() == 'c':
            price[i] = max(sim[i,-1] - K, 0)
        else:
            price[i] = max(K - sim[i,-1], 0)
            
    return np.mean(price) * dis_factor

'''
=============================
Antithetic Variable Technique
=============================
'''
def AVT(optype, S, K, T, r, sig, path, steps):
    dt = T/steps
    rands = np.random.normal(0,1,(path*steps,1))
    rands = np.append(rands,-rands).reshape(2*len(rands),1)
    
    sim = np.zeros((2*path,steps+1))
    
    sim[:,0] = S
    
    for i in range(2*path):
        for j in range(1,steps+1):
            sim[i,j] = sim[i,j-1] * np.exp((r-pow(sig,2)/2)*dt+sig*np.sqrt(dt)*rands[i*steps+j-1,0])
            if sim[i,j] < H:
                break
    
    price = np.zeros(sim.shape[0])
    for i in range(sim.shape[0]):
        if sim[i,-1] == 0:
            price[i] = 0
        elif optype.lower() == 'c':
            price[i] = max(sim[i,-1] - K, 0)
        else:
            price[i] = max(K - sim[i,-1], 0)
 
    return np.mean(price) * dis_factor

def MC_option(optype, simtype, S, K, T, r, sig, path, steps):
    
    if simtype.lower() == 'mcs':
        sim = MCS(S, K, T, r, sig, path, steps)
    elif simtype.lower() == 'avt':
        sim = AVT(S, K, T, r, sig, path, steps)
    
    price = np.zeros(sim.shape[0])
    for i in range(sim.shape[0]):
        if min(sim[i,:]) < H:
            price[i] = 0
        elif optype.lower() == 'c':
            price[i] = max(sim[i,-1] - K, 0)
        else:
            price[i] = max(K - sim[i,-1], 0)
    
    return np.mean(price)

'''
=============================
Control Variate Technique
=============================
'''
def CVT(optype, S, K, T, r, sig, path, steps):
    optype = optype.lower()
    
    '''BS price'''
    bs_price = bs.black_scholes('c', S, K, T, r, sig)
    
    '''MC price'''
    mc_sim = MCS(S, K, T, r, sig, path, 1)
    
    '''Barrier option price using MC'''
    brr_price = MC_option(optype, 'mcs', S, K, T, r, sig, path, steps)
    
    if optype == 'c':
        mc_price = np.mean(np.maximum(mc_sim[:,-1] - K, 0))
        price = brr_price + (bs_price - mc_price)
    
    else:
        mc_price = np.mean(K - np.maximum(mc_sim[:,-1], 0))
        price = brr_price + (bs_price - mc_price)

    return price


def BS(S, X, T, r, sigma):
    
    result = {}
    s = time.time()
    result['Call'] = [bs.black_scholes('c', S, X, T, r, sigma)]
    result['Put'] = [bs.black_scholes('p', S, X, T, r, sigma)]
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

result = pd.DataFrame(BS(S0, K, T, r, sig))

Path = [100, 1000, 10000]#, 100000, 1000000]
for k ,path in enumerate(Path):
    mc_p = np.zeros((rep,len(result.columns)))
    avt_p = np.zeros((rep,len(result.columns)))
    cvt_p = np.zeros((rep,len(result.columns)))
    
    for i in range(rep):
        '''crude MC'''
        s1 = time.time()
        mc_call = MC_option('c', 'mcs', S0, K, T, r, sig, path, 1)
        mc_put = MC_option('p', 'mcs', S0, K, T, r, sig, path, 1)
        t1 = time.time() - s1
        
        mc_call_s1 = MC_option('c', 'mcs', S0*(1+d), K, T, r, sig, path, 1)
        mc_call_s2 = MC_option('c', 'mcs', S0*(1-d), K, T, r, sig, path, 1)
        
        # delta
        mc_delta_call = (mc_call_s1 - mc_call_s2) / (2*S0*d)
        mc_delta_put = (MC_option('p', 'mcs', S0*(1+d), K, T, r, sig, path, 1) - MC_option('p', 'mcs', S0*(1-d), K, T, r, sig, path, 1)) / (2*S0*d)
        
        # gamma
        mc_gamma = (mc_call_s1 - 2*mc_call + mc_call_s2) / pow(d*S0,2)
        
        # vega
        mc_vega = (MC_option('c', 'mcs', S0, K, T, r, sig*(1+d), path, 1) - MC_option('c', 'mcs', S0, K, T, r, sig*(1-d), path, 1)) / (2*d*sig)
        
        # rho
        mc_rho_call = (MC_option('c', 'mcs', S0, K, T, r*(1+d), sig, path, 1) - MC_option('c', 'mcs', S0, K, T, r*(1-d), sig, path, 1)) / (2*d*r)
        mc_rho_put = (MC_option('p', 'mcs', S0, K, T, r*(1+d), sig, path, 1) - MC_option('p', 'mcs', S0, K, T, r*(1-d), sig, path, 1)) / (2*d*r)
        
        # theta
        mc_theta_call = (MC_option('c', 'mcs', S0, K, T*(1+d), r, sig, path, 1) - MC_option('c', 'mcs', S0, K, T*(1+d), r, sig, path, 1)) / (2*d*T)
        mc_theta_put = (MC_option('p', 'mcs', S0, K, T*(1+d), r, sig, path, 1) - MC_option('p', 'mcs', S0, K, T*(1+d), r, sig, path, 1)) / (2*d*T)
        
        mc_p[i,:] = [mc_call, mc_put, mc_delta_call, mc_delta_put, mc_gamma, mc_vega, mc_rho_call, mc_rho_put, mc_theta_call, mc_theta_put, t1]
        
        '''AVT'''
        s2 = time.time()
        avt_call = MC_option('c', 'avt', S0, K, T, r, sig, path, steps)
        avt_put = MC_option('p', 'avt', S0, K, T, r, sig, path, steps)
        t2 = time.time() - s2
        
        avt_call_s1 = MC_option('c', 'avt', S0*(1+d), K, T, r, sig, path, steps)
        avt_call_s2 = MC_option('c', 'avt', S0*(1-d), K, T, r, sig, path, steps)
        
        # delta
        avt_delta_call = (avt_call_s1 - avt_call_s2) / (2*S0*d)
        avt_delta_put = (MC_option('p', 'avt', S0*(1+d), K, T, r, sig, path, steps) - MC_option('p', 'avt', S0*(1-d), K, T, r, sig, path, steps)) / (2*S0*d)
        
        # gamma
        avt_gamma = (avt_call_s1 - 2*avt_call + avt_call_s2) / pow(d*S0,2)
        
        # vega
        avt_vega = (MC_option('c', 'avt', S0, K, T, r, sig*(1+d), path, steps) - MC_option('c', 'avt', S0, K, T, r, sig*(1-d), path, steps)) / (2*d*sig)
        
        # rho
        avt_rho_call = (MC_option('c', 'avt', S0, K, T, r*(1+d), sig, path, steps) - MC_option('c', 'avt', S0, K, T, r*(1-d), sig, path, steps)) / (2*d*r)
        avt_rho_put = (MC_option('p', 'avt', S0, K, T, r*(1+d), sig, path, steps) - MC_option('p', 'avt', S0, K, T, r*(1-d), sig, path, steps)) / (2*d*r)
        
        # theta
        avt_theta_call = (MC_option('c', 'avt', S0, K, T*(1+d), r, sig, path, 1) - MC_option('c', 'avt', S0, K, T*(1+d), r, sig, path, steps)) / (2*d*T)
        avt_theta_put = (MC_option('p', 'avt', S0, K, T*(1+d), r, sig, path, 1) - MC_option('p', 'avt', S0, K, T*(1+d), r, sig, path, steps)) / (2*d*T)
        
        avt_p[i,:] = [avt_call, avt_put, avt_delta_call, avt_delta_put, avt_gamma, avt_vega, avt_rho_call, avt_rho_put, avt_theta_call, avt_theta_put, t2]
        
        '''CVT'''
        s3 = time.time()
        cvt_call = CVT('c', S0, K, T, r, sig, path, steps)
        cvt_put = CVT('p', S0, K, T, r, sig, path, steps)
        t3 = time.time() - s3
         
        cvt_call_s1 = CVT('c', S0*(1+d), K, T, r, sig, path, steps)
        cvt_call_s2 = CVT('c', S0*(1-d), K, T, r, sig, path, steps)
        
        # delta
        cvt_delta_call = (cvt_call_s1 - cvt_call_s2) / (2*S0*d)
        cvt_delta_put = (CVT('p', S0*(1+d), K, T, r, sig, path, steps) - CVT('p', S0*(1-d), K, T, r, sig, path, steps)) / (2*S0*d)
        
        # gamma
        cvt_gamma = (cvt_call_s1 - 2*cvt_call + cvt_call_s2) / pow(d*S0,2)
        
        # vega
        cvt_vega = (CVT('c', S0, K, T, r, sig*(1+d), path, steps) - CVT('c', S0, K, T, r, sig*(1-d), path, steps)) / (2*d*sig)
        
        # rho
        cvt_rho_call = (CVT('c', S0, K, T, r*(1+d), sig, path, steps) - CVT('c', S0, K, T, r*(1-d), sig, path, steps)) / (2*d*r)
        cvt_rho_put = (CVT('p', S0, K, T, r*(1+d), sig, path, steps) - CVT('p', S0, K, T, r*(1-d), sig, path, steps)) / (2*d*r)
        
        # theta
        cvt_theta_call = (CVT('c', S0, K, T*(1+d), r, sig, path, 1) - CVT('c', S0, K, T*(1+d), r, sig, path, steps)) / (2*d*T)
        cvt_theta_put = (CVT('p', S0, K, T*(1+d), r, sig, path, 1) - CVT('p', S0, K, T*(1+d), r, sig, path, steps)) / (2*d*T)
        
        cvt_p[i,:] = [cvt_call, cvt_put, cvt_delta_call, cvt_delta_put, cvt_gamma, cvt_vega, cvt_rho_call, cvt_rho_put, cvt_theta_call, cvt_theta_put, t3]
        
    result.loc[k*6+1,:] = np.mean(mc_p, axis=0)
    result.loc[k*6+2,:] = np.std(mc_p, axis=0)
    result.loc[k*6+3,:] = np.mean(avt_p, axis=0)
    result.loc[k*6+4,:] = np.std(avt_p, axis=0)
    result.loc[k*6+5,:] = np.mean(cvt_p, axis=0)
    result.loc[k*6+6,:] = np.std(cvt_p, axis=0)


