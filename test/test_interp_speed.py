#%%
import numpy as np
import sys
sys.path.append('..')
from util.sdtpy import *
from util.helper import *
#%%
import pickle
with open('../model/gw170817-like.interp.pkl', 'rb') as f:
	interp = pickle.load(f)
#%%
# mdarr = np.array([0.001, 0.003, 0.01, 0.03, 0.1])
# vdarr = np.array([0.05, 0.15, 0.3])
# mwarr = np.array([0.001, 0.003, 0.01, 0.03, 0.1])
# vwarr = np.array([0.05, 0.15, 0.3])

# lamarr = np.arange(1003, 127695.+1, 1)
lamarr = np.arange(3000, 10000.+1, 1)

point = (
	0.05,
	0.1,
	0.05,
	0.1,
	45,
	0.5,
	lamarr,
)
# %%
%%timeit
interp(point)
# %%
