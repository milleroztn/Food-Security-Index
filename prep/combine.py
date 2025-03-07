import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from numpy import nan as NA
from fao_explore import which_areas

gfsi_r = (pd.read_csv("../data/gfsi_r.csv").pivot(index = 'Year', columns = 'Area')
  .rename(columns={'SSA Average (mean)': 'mean', 'SSA Average (median)': 'median'})
  .loc[:,pd.IndexSlice[:, ['mean','median']]]
  )

gfsi_r = pd.concat({'SSA': gfsi_r}, names=['Area'])
gfsi_r.columns = gfsi_r.columns.map('_'.join).str.strip('_')

FAO_r = pd.read_csv("../data/FAO_r.csv").set_index(['Area','Year'])

gfsi_r.join(FAO_r, how='right').to_csv("../data/data_r.csv")


gfsi_SSA = pd.read_csv("../data/gfsi_SSA.csv").set_index(['Area','Year'])
FAO_SSA = pd.read_csv("../data/FAO_SSA.csv").set_index(['Area','Year'])

# which_areas(FAO_SSA,gfsi_SSA)

gfsi_SSA = (gfsi_SSA.reset_index().replace({'Congo (Dem. Rep.)': 'Democratic Republic of the Congo', 
'Tanzania': 'United Republic of Tanzania'}).set_index(['Area','Year'])
)

gfsi_SSA.join(FAO_SSA, how='right').to_csv("../data/data_SSA.csv")
