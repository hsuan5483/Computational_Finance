#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Pei-Hsuan Hsu
"""

import numpy as np
from scipy.stats import norm

'''
params                                                    type
======                                                    =====
S: underlying futures price                               float
K: strike price                                           float
sigma: annualized standard deviation, or volatility       float
t: time to expiration in years                            float
r: risk-free interest rate                                float
'''

def d1(S, K, t, r, sigma): # keep r argument for consistency
    
    numerator = np.log(S/K) + (r + 0.5 * pow(sigma,2))*t
    denominator = sigma*np.sqrt(t)

    return numerator/denominator

def d2(S, K, t, r, sigma): # keep r argument for consistency
        
    return d1(S, K, t, r, sigma) - sigma*np.sqrt(t)

def black_scholes(flag, S, K, t, r, sigma):
    flag = flag.lower()
    disfac = np.exp(-r*t)
    D1 = d1(S, K, t, r, sigma)
    D2 = d2(S, K, t, r, sigma)
    
    if flag == 'c':
        price = S * norm.cdf(D1,0,1) - K * disfac * norm.cdf(D2,0,1)

    else:
        price = - S * norm.cdf(-D1,0,1) + K * disfac * norm.cdf(-D2,0,1)

    return price

def delta(flag, S, K, t, r, sigma):
    '''
    公式：dC/dS or dP/dS
    買(賣)權的delta：dS增加一單位，買(賣)權價值增加(減少)delta單位
    '''
    flag = flag.lower()
    d_1 = d1(S, K, t, r, sigma)

    if flag == 'p':
        return norm.cdf(d_1) - 1.0
    else:
        return norm.cdf(d_1)

def theta(flag, S, K, t, r, sigma):
    flag = flag.lower()
    two_sqrt_t = 2 * np.sqrt(t)

    D1 = d1(S, K, t, r, sigma)
    D2 = d2(S, K, t, r, sigma)

    first_term = (-S * norm.pdf(D1,0,1) * sigma) / two_sqrt_t 

    if flag == 'c':

        second_term = r * K * np.exp(-r*t) * norm.cdf(D2,0,1)
        return (first_term - second_term)/365.0
    
    if flag == 'p':
    
        second_term = r * K * np.exp(-r*t) * norm.cdf(-D2,0,1)
        return (first_term + second_term)/365.0

def gamma(flag, S, K, t, r, sigma):
    flag = flag.lower()
    d_1 = d1(S, K, t, r, sigma)
    return norm.pdf(d_1)/(S*sigma*np.sqrt(t))

def vega(flag, S, K, t, r, sigma):
    flag = flag.lower()
    d_2 = d2(S, K, t, r, sigma)
    disfac = np.exp(-r*t)
    if flag == 'c':
        return t*K*disfac * norm.cdf(d_2,0,1) * .01
    else:
        return -t*K*disfac * norm.cdf(-d_2,0,1) * .01

def rho(flag, S, K, t, r, sigma):
    flag = flag.lower()
    d_2 = d2(S, K, t, r, sigma)
    disfac = np.exp(-r*t)
    if flag == 'c':
        return t*K*disfac * norm.cdf(d_2,0,1) * .01
    else:
        return -t*K*disfac * norm.cdf(-d_2,0,1) * .01

