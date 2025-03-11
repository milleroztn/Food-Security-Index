import pandas as pd
import numpy as np
from numpy import nan as NA

GFSI_loc = './raw data/GFSI/'
clean_loc = '../data/'

### Import and concatentate each year into single file
gfsi = pd.DataFrame()

def import_GFSI(y):
    raw = pd.read_excel(GFSI_loc+str(y)+'_SSA.xlsx', skiprows=6, usecols='B:AI', index_col=0).T
    raw['Year'] = y
    return raw.reset_index(names='Area').set_index(['Area','Year'])

for i in range(2012,2023):
  raw = import_GFSI(i)
  gfsi = pd.concat([gfsi,raw]).sort_index()
  
 
### identify and rename main 5 index elements
gfsi = gfsi.rename(columns = {'FOOD SECURITY ENVIRONMENT':'fs', '1) AFFORDABILITY':'afford', '2) AVAILABILITY':'avail', '3) QUALITY AND SAFETY':'qual', '4) SUSTAINABILITY AND ADAPTATION':'adapt'})


### entire file
gfsi.to_csv(clean_loc+'gfsi_all.csv')

### only main 5 index elements
gfsi = gfsi.iloc[:,[0,1,13,46,65]]
gfsi.to_csv(clean_loc+'gfsi_main5.csv')

### only regional summary stats per year
gfsi_r = gfsi.reset_index()[gfsi.reset_index().Area.isin(['Average (mean)','Average (median)','Maximum','Minimum'])]
gfsi_r['Area'] = gfsi_r['Area'].apply(lambda x: 'SSA '+x)
gfsi_r.to_csv(clean_loc+'gfsi_r.csv', index=False)

### index values for each country per year
gfsi_ssa = gfsi.reset_index()[~(gfsi.reset_index().Area.isin(['Average (mean)','Average (median)','Maximum','Minimum','Weight']))]
gfsi_ssa.to_csv('../data/gfsi_SSA.csv', index=False)



