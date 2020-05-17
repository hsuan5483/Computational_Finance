#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Pei-Hsuan Hsu
"""

import numpy as np
import pandas as pd
from scipy.misc import derivative
import time
import Black_Scholes as bs

def MCS(S0, K, r, T, sigma, paths, steps, output = False, outfilename = None, rep=1):
    np.random.seed(123457)
    delta = T/steps
    rands = np.random.normal(0,1,(steps,paths))
    
    simulation = np.zeros((steps+1,paths))
    simulation[0,:] = S0
    
    results = []
    for i in range(paths):
        for j in range(steps):
            
            simulation[j+1, i] = simulation[j, i] *\
            np.exp((r - 0.5 * pow(sigma, 2)) * delta + sigma * np.sqrt(delta) * rands[j,i])

        results.append(simulation[-1,i])
    
    if output:
        data = simulation[:,:30]
        plot = data.plot.line(legend=None)
        fig = plot.get_figure()
        fig.savefig(outfilename)
    
    return results

def MCS_Options(optype, S0, K, r, T, sigma, d, paths, steps, discount, output = False, outfilename = None, rep=1):
    s = time.time()
    ST = MCS(S0, K, r, T, sigma, paths, T/steps, output = False, outfilename = None, rep=1)
    
    call = []
    put = []
    calldelta = []
    putdelta = []
    calltheta = []
    puttheta = []
    gamma = []
    vega = []
    callrho = []
    putrho = []
    
    for S in ST:
        call.append(bs.black_scholes('c', S, K, T, r, sigma, discount))
        put.append(bs.black_scholes('p', S, K, T, r, sigma, discount))
        calldelta.append(bs.delta('c', S, K, T, r, sigma))
        putdelta.append(bs.delta('p', S, K, T, r, sigma))
        gamma.append(bs.gamma('c', S, K, T, r, sigma))
        vega.append(bs.vega('c', S, K, T, r, sigma))
        callrho.append(bs.rho('c', S, K, T, r, sigma))
        putrho.append(bs.rho('p', S, K, T, r, sigma))
        calltheta.append(bs.theta('c', S, K, T, r, sigma))
        puttheta.append(bs.theta('p', S, K, T, r, sigma))

    call = np.mean(call)
    put = np.mean(put)
    calldelta = np.mean(calldelta)
    putdelta = np.mean(putdelta)
    calltheta = np.mean(calltheta)
    puttheta = np.mean(puttheta)
    gamma = np.mean(gamma)
    vega = np.mean(vega)
    callrho = np.mean(callrho)
    putrho = np.mean(putrho)
    t = time.time()-s
    
    return call, put, calldelta, putdelta, gamma, vega, callrho, putrho, calltheta, puttheta, t
