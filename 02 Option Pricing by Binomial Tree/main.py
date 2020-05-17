#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Pei-Hsuan Hsu
"""

import os
from os.path import join
import numpy as np
import pandas as pd
import Black_Scholes as bs
import CRR
import time
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt

path = os.getcwd()
out = join(path, 'output')

'''parameters'''
S = 55
X = 60
r = 0.05
sigma = 0.4
T = 0.5
n = 5
d = 0.01

'''
===================
Black-Scholes Model
===================
'''
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

result = pd.DataFrame(BS(S, X, T, r, sigma))

'''
=========
CRR Model
=========
'''
### Greeks using one tree
n_list = [5, 10, 25, 50, 75, 100, 250, 500, 1000]#, 5000, 10000]

'''European'''
i = 0
print('European Option')
for n in n_list:
    i += 1
    print('Running n=',n,' ...')
    for otype in ['c','p']:
        
        s = time.time()
        if otype == 'c':
            stockvalue, optionvalue,_ = CRR.EurCRR(n, S, X, r, sigma, T, otype)
            
            result.loc[i,'Call'] = optionvalue[0,0]
            
            # delta
            result.loc[i,'Delta_Call'] = (optionvalue[0,1] - optionvalue[1,1])/(stockvalue[0,1] - stockvalue[1,1])
            
            # gamma
            d1 = (optionvalue[0,2] - optionvalue[1,2])/(stockvalue[0,2] - stockvalue[1,2])
            d2 = (optionvalue[1,2] - optionvalue[2,2])/(stockvalue[1,2] - stockvalue[2,2])
            result.loc[i,'Gamma'] = (d1-d2)/(stockvalue[0,1] - stockvalue[1,1])
            
            # theta
            result.loc[i,'Theta_Call'] = (optionvalue[1,2] - optionvalue[0,0])/(2*T/n)
            
        else:
            stockvalue, optionvalue,_ = CRR.EurCRR(n, S, X, r, sigma, T, otype)
            
            result.loc[i,'Put'] = optionvalue[0,0]
            
            # delta
            result.loc[i,'Delta_Put'] = (optionvalue[0,1] - optionvalue[1,1])/(stockvalue[0,1] - stockvalue[1,1])
                    
            # theta
            result.loc[i,'Theta_Put'] = (optionvalue[1,2] - optionvalue[0,0])/(2*T/n)
    
    e = time.time() - s
    result.loc[i,'CPU time'] = e
    print('time=',e)

'''American'''
print('American Option')
for n in n_list:
    i += 1
    print('Running n=',n,' ...')
    for otype in ['c','p']:
        s = time.time()
        if otype == 'c':
            stockvalue, optionvalue,_ = CRR.AmerCRR(n, S, X, r, sigma, T, otype)
            
            result.loc[i,'Call'] = optionvalue[0,0]
            
            # delta
            result.loc[i,'Delta_Call'] = (optionvalue[0,1] - optionvalue[1,1])/(stockvalue[0,1] - stockvalue[1,1])
            
            # gamma
            d1 = (optionvalue[0,2] - optionvalue[1,2])/(stockvalue[0,2] - stockvalue[1,2])
            d2 = (optionvalue[1,2] - optionvalue[2,2])/(stockvalue[1,2] - stockvalue[2,2])
            result.loc[i,'Gamma'] = (d1-d2)/(stockvalue[0,1] - stockvalue[1,1])
            
            # theta
            result.loc[i,'Theta_Call'] = (optionvalue[1,2] - optionvalue[0,0])/(2*T/n)
            
        else:
            stockvalue, optionvalue,_ = CRR.AmerCRR(n, S, X, r, sigma, T, otype)
            
            result.loc[i,'Put'] = optionvalue[0,0]
            
            # delta
            result.loc[i,'Delta_Put'] = (optionvalue[0,1] - optionvalue[1,1])/(stockvalue[0,1] - stockvalue[1,1])
                    
            # theta
            result.loc[i,'Theta_Put'] = (optionvalue[1,2] - optionvalue[0,0])/(2*T/n)
    
    e = time.time() - s
    result.loc[i,'CPU time'] = e
    print('time=',e)

result.index = ['BS'] + n_list + n_list
result.to_excel(join(out, 'result.xlsx'), sheet_name = 'one tree')

writer = pd.ExcelWriter(join(out,'result.xlsx'), engine='openpyxl', mode='a')

### Greeks using two trees
result2 = pd.DataFrame(BS(S, X, T, r, sigma))
'''European'''
i = 0
print('European Option')
for n in n_list:
    i += 1
    print('Running n=',n,' ...')
    for otype in ['c','p']:
        s = time.time()
        
        if otype == 'c':
            '''simulation'''
            stockvalue, optionvalue,_ = CRR.EurCRR(n, S, X, r, sigma, T, otype)
            
            stockvalue_s1, optionvalue_s1,_ = CRR.EurCRR(n, S*(1+d), X, r, sigma, T, otype)
            stockvalue_s2, optionvalue_s2,_ = CRR.EurCRR(n, S*(1-d), X, r, sigma, T, otype)
            
            # theta
            _, optionvalue_t1,_ = CRR.EurCRR(n, S, X, r, sigma, T*(1+d), otype)
            _, optionvalue_t2,_ = CRR.EurCRR(n, S, X, r, sigma, T*(1-d), otype)
            
            # vega
            _, optionvalue_sig1,_ = CRR.EurCRR(n, S, X, r, sigma*(1+d), T, otype)
            _, optionvalue_sig2,_ = CRR.EurCRR(n, S, X, r, sigma*(1-d), T, otype)
            
            # rho
            _, optionvalue_r1,_ = CRR.EurCRR(n, S, X, r*(1+d), sigma, T, otype)
            _, optionvalue_r2,_ = CRR.EurCRR(n, S, X, r*(1-d), sigma, T, otype)

            '''calculation'''
            result2.loc[i,'Call'] = optionvalue[0,0]
            
            # delta
            result2.loc[i,'Delta_Call'] = (optionvalue_s1[0,0] - optionvalue_s2[0,0])/(2*d*S)
            
            # gamma
            result2.loc[i,'Gamma'] = (optionvalue_s1[0,0] - 2*optionvalue[0,0] + optionvalue_s2[0,0])/pow(d*S,2)
            
            # theta
            result2.loc[i,'Theta_Call'] = (optionvalue_t2[0,0] - optionvalue_t1[0,0])/(2*d*T)
            
            # vega
            result2.loc[i,'Vega'] = (optionvalue_sig2[0,0] - optionvalue_sig1[0,0])/(2*d*sigma)
            
            # rho
            result2.loc[i,'Rho_Call'] = (optionvalue_r1[0,0] - optionvalue_r2[0,0])/(2*d*r)
            
        else:
            '''simulation'''
            stockvalue, optionvalue,_ = CRR.EurCRR(n, S, X, r, sigma, T, otype)
            
            _, optionvalue_s1,_ = CRR.EurCRR(n, S*(1+d), X, r, sigma, T, otype)
            _, optionvalue_s2,_ = CRR.EurCRR(n, S*(1-d), X, r, sigma, T, otype)
            
            # theta
            _, optionvalue_t1,_ = CRR.EurCRR(n, S, X, r, sigma, T*(1+d), otype)
            _, optionvalue_t2,_ = CRR.EurCRR(n, S, X, r, sigma, T*(1-d), otype)
            
            # rho
            _, optionvalue_r1,_ = CRR.EurCRR(n, S, X, r*(1+d), sigma, T, otype)
            _, optionvalue_r2,_ = CRR.EurCRR(n, S, X, r*(1-d), sigma, T, otype)

            '''calculation'''
            result2.loc[i,'Put'] = optionvalue[0,0]
            
            # delta
            result2.loc[i,'Delta_Put'] = (optionvalue_s1[0,0] - optionvalue_s2[0,0])/(2*d*S)
                        
            # theta
            result2.loc[i,'Theta_Put'] = (optionvalue_t2[0,0] - optionvalue_t1[0,0])/(2*d*T)
            
            # rho
            result2.loc[i,'Rho_Put'] = (optionvalue_r1[0,0] - optionvalue_r2[0,0])/(2*d*r)
            
    e = time.time() - s
    result2.loc[i,'CPU time'] = e
    print('time=',e)


'''American'''
print('European Option')
for n in n_list:
    i += 1
    print('Running n=',n,' ...')
    for otype in ['c','p']:
        s = time.time()
        
        if otype == 'c':
            '''simulation'''
            stockvalue, optionvalue,_ = CRR.AmerCRR(n, S, X, r, sigma, T, otype)
            
            stockvalue_s1, optionvalue_s1,_ = CRR.AmerCRR(n, S*(1+d), X, r, sigma, T, otype)
            stockvalue_s2, optionvalue_s2,_ = CRR.AmerCRR(n, S*(1-d), X, r, sigma, T, otype)
            
            # theta
            _, optionvalue_t1,_ = CRR.AmerCRR(n, S, X, r, sigma, T*(1+d), otype)
            _, optionvalue_t2,_ = CRR.AmerCRR(n, S, X, r, sigma, T*(1-d), otype)
            
            # vega
            _, optionvalue_sig1,_ = CRR.AmerCRR(n, S, X, r, sigma*(1+d), T, otype)
            _, optionvalue_sig2,_ = CRR.AmerCRR(n, S, X, r, sigma*(1-d), T, otype)
            
            # rho
            _, optionvalue_r1,_ = CRR.AmerCRR(n, S, X, r*(1+d), sigma, T, otype)
            _, optionvalue_r2,_ = CRR.AmerCRR(n, S, X, r*(1-d), sigma, T, otype)

            '''calculation'''
            result2.loc[i,'Call'] = optionvalue[0,0]
            
            # delta
            result2.loc[i,'Delta_Call'] = (optionvalue_s1[0,0] - optionvalue_s2[0,0])/(2*d*S)
            
            # gamma
            result2.loc[i,'Gamma'] = (optionvalue_s1[0,0] - 2*optionvalue[0,0] + optionvalue_s2[0,0])/pow(d*S,2)
            
            # theta
            result2.loc[i,'Theta_Call'] = (optionvalue_t2[0,0] - optionvalue_t1[0,0])/(2*d*T)
            
            # vega
            result2.loc[i,'Vega'] = (optionvalue_sig1[0,0] - optionvalue_sig2[0,0])/(2*d*sigma)
            
            # rho
            result2.loc[i,'Rho_Call'] = (optionvalue_r1[0,0] - optionvalue_r2[0,0])/(2*d*r)
            
        else:
            '''simulation'''
            stockvalue, optionvalue,_ = CRR.AmerCRR(n, S, X, r, sigma, T, otype)
            
            _, optionvalue_s1,_ = CRR.AmerCRR(n, S*(1+d), X, r, sigma, T, otype)
            _, optionvalue_s2,_ = CRR.AmerCRR(n, S*(1-d), X, r, sigma, T, otype)
            
            # theta
            _, optionvalue_t1,_ = CRR.AmerCRR(n, S, X, r, sigma, T*(1+d), otype)
            _, optionvalue_t2,_ = CRR.AmerCRR(n, S, X, r, sigma, T*(1-d), otype)
            
            # rho
            _, optionvalue_r1,_ = CRR.AmerCRR(n, S, X, r*(1+d), sigma, T, otype)
            _, optionvalue_r2,_ = CRR.AmerCRR(n, S, X, r*(1-d), sigma, T, otype)

            '''calculation'''
            result2.loc[i,'Put'] = optionvalue[0,0]
            
            # delta
            result2.loc[i,'Delta_Put'] = (optionvalue_s1[0,0] - optionvalue_s2[0,0])/(2*d*S)
                        
            # theta
            result2.loc[i,'Theta_Put'] = (optionvalue_t2[0,0] - optionvalue_t1[0,0])/(2*d*T)
            
            # rho
            result2.loc[i,'Rho_Put'] = (optionvalue_r1[0,0] - optionvalue_r2[0,0])/(2*d*r)
            
    e = time.time() - s
    result2.loc[i,'CPU time'] = e
    print('time=',e)
    
result2.index = ['BS'] + n_list + n_list
result2.to_excel(writer, sheet_name = 'two trees')

writer.save()


'''
===========
Plot Greeks
===========
'''
N = 300
df_plot = pd.DataFrame(columns = result.columns[2:-1])
for otype in ['c','p']:
    
    for n in range(1,N+1):
        
        if otype == 'c':
            '''simulation'''
            stockvalue, optionvalue,_ = CRR.EurCRR(n, S, X, r, sigma, T, otype)
            
            stockvalue_s1, optionvalue_s1,_ = CRR.EurCRR(n, S*(1+d), X, r, sigma, T, otype)
            stockvalue_s2, optionvalue_s2,_ = CRR.EurCRR(n, S*(1-d), X, r, sigma, T, otype)
            
            # theta
            _, optionvalue_t1,_ = CRR.EurCRR(n, S, X, r, sigma, T*(1+d), otype)
            _, optionvalue_t2,_ = CRR.EurCRR(n, S, X, r, sigma, T*(1-d), otype)
            
            # vega
            _, optionvalue_sig1,_ = CRR.EurCRR(n, S, X, r, sigma*(1+d), T, otype)
            _, optionvalue_sig2,_ = CRR.EurCRR(n, S, X, r, sigma*(1-d), T, otype)
            
            # rho
            _, optionvalue_r1,_ = CRR.EurCRR(n, S, X, r*(1+d), sigma, T, otype)
            _, optionvalue_r2,_ = CRR.EurCRR(n, S, X, r*(1-d), sigma, T, otype)

            '''calculation'''            
            # delta
            df_plot.loc[n-1,'Delta_Call'] = (optionvalue_s1[0,0] - optionvalue_s2[0,0])/(2*d*S)
            
            # gamma
            df_plot.loc[n-1,'Gamma'] = (optionvalue_s1[0,0] - 2*optionvalue[0,0] + optionvalue_s2[0,0])/pow(d*S,2)
            
            # theta
            df_plot.loc[n-1,'Theta_Call'] = (optionvalue_t2[0,0] - optionvalue_t1[0,0])/(2*d*T)
            
            # vega
            df_plot.loc[n-1,'Vega'] = (optionvalue_sig1[0,0] - optionvalue_sig2[0,0])/(2*d*sigma)
            
            # rho
            df_plot.loc[n-1,'Rho_Call'] = (optionvalue_r1[0,0] - optionvalue_r2[0,0])/(2*d*r)
            
        else:
            '''simulation'''
            stockvalue, optionvalue,_ = CRR.EurCRR(n, S, X, r, sigma, T, otype)
            
            _, optionvalue_s1,_ = CRR.EurCRR(n, S*(1+d), X, r, sigma, T, otype)
            _, optionvalue_s2,_ = CRR.EurCRR(n, S*(1-d), X, r, sigma, T, otype)
            
            # theta
            _, optionvalue_t1,_ = CRR.EurCRR(n, S, X, r, sigma, T*(1+d), otype)
            _, optionvalue_t2,_ = CRR.EurCRR(n, S, X, r, sigma, T*(1-d), otype)
            
            # rho
            _, optionvalue_r1,_ = CRR.EurCRR(n, S, X, r*(1+d), sigma, T, otype)
            _, optionvalue_r2,_ = CRR.EurCRR(n, S, X, r*(1-d), sigma, T, otype)

            '''calculation'''
            # delta
            df_plot.loc[n-1,'Delta_Put'] = (optionvalue_s1[0,0] - optionvalue_s2[0,0])/(2*d*S)
                        
            # theta
            df_plot.loc[n-1,'Theta_Put'] = (optionvalue_t2[0,0] - optionvalue_t1[0,0])/(2*d*T)
            
            # rho
            df_plot.loc[n-1,'Rho_Put'] = (optionvalue_r1[0,0] - optionvalue_r2[0,0])/(2*d*r)
        

# matplotlib
fig , ax = plt.subplots()
for i, col in enumerate(df_plot.columns):
    plt.subplot(2, 4, i+1)
    plt.plot(df_plot.index, df_plot[col])
    plt.title(col)
    plt.xlabel('stage', fontsize = 18)
    
plt.savefig(join(out, 'Greeks.png'))  


# plotly
fig = make_subplots(
    rows=2, cols=4,
    subplot_titles=df_plot.columns)

for i, col in enumerate(df_plot.columns):
    if i < 4:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[col], name = col, mode = 'lines'),
              row=1, col=i+1)
        
    else:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[col], name = col, mode = 'lines'),
              row=2, col=i-3)

fig.update_layout(height=800, width=1800,
                  title_text="Greeks")

plotly.offline.plot(fig , filename = join(out, 'Greeks.html'))
