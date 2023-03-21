import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series, DataFrame
from numpy import nan as NA

gfsi_r = pd.read_csv("../data/gfsi_r.csv").pivot(index = 'Year', columns = 'Area')
FAO_r = pd.read_csv("../data/FAO_r.csv").pivot(index = 'Year', columns = 'Area')


gfsi_r.join(FAO_r, how='right').to_csv("../data/data_r_wide.csv", index=False)
