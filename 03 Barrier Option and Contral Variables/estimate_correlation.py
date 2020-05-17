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

home = os.getcwd()
src = join(home, 'source')
out = join(home, 'output')

N = 20000
dim = 5

corr = np.array(pd.read_excel(join(src, "corr.xlsx"), index_col=0))
upper_cho = cholesky(corr)
rands = np.random.normal(0,1,(N,dim))

corr_rands = pd.DataFrame(np.inner(rands,upper_cho))

rnd_corr = corr_rands.corr(method='pearson')
rnd_corr.to_excel(join(out, 'rand corr.xlsx'))
