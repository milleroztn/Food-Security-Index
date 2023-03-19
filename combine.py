import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series, DataFrame
from numpy import nan as NA
import xgboost as xgb

gfsi_r = pd.read_csv("../data/gfsi_r.csv")
FAO_r = pd.read_csv("../data/FAO_r.csv")



# gfsi = gfsi_r.set_index('year')
# 
# gfsi[gfsi.area == 'ssa average (mean)'].plot()
# 
# gfsi[gfsi.area == 'ssa average (median)'].plot()
# 
# plt.show()
# 
# gfsi.plot()
