import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from numpy import nan as NA
import faostat


# Function to calculate weighted average
def weighted_average(dataframe, value, weight):
    val = dataframe[value]
    wt = dataframe[weight]
    return (val * wt).sum() / wt.sum()

# Function to summarize which variables are available for which years in which country
def which_years(x):

    # Initialize an empty list to hold the rows of the output table
    rows = []
    
    # Loop through each region in the dataset
    for region in x.reset_index()['Area'].unique():
        
        # Loop through each variable in the dataset
        for variable in x.columns:
            years_not_missing = []
            
            # Loop through each year for the current region and variable
            for year in x.reset_index().loc[(x.reset_index()['Area'] == region), 'Year']:
                value = x.reset_index().loc[(x.reset_index()['Area'] == region) & (x.reset_index()['Year'] == year), variable].values[0]
                
                # Check if the value is not missing
                if pd.notna(value):
                    years_not_missing.append(year)
                    
            # Calculate the range of years with data for the current variable in the current region
            if len(years_not_missing) > 0:
                min_year = min(years_not_missing)
                max_year = max(years_not_missing)
                num_missing = (int(max_year) - int(min_year) + 1) - len(years_not_missing)
            else:
                min_year = 'None'
                max_year = 'None'
                num_missing = 'None'
                    
            # Add a row to the output table for the current region and variable
            row = {
                'Region': region,
                'Variable': variable,
                'Min Year': min_year,
                'Max Year': max_year,
                'Num Years Missing': num_missing
            }
            rows.append(row)
            
    # Output the summary table
    return pd.DataFrame(rows)
  
# Define function to transform column names
def transform_colname(name,x):
    # Split the column name by whitespace and take the first word
    first_word = name.split()[0]
    
    # Remove comma from end of first word, if present
    if first_word[-1] == ',':
        first_word = first_word[:-1]
    
    # Add 'prod' or whatever before first word and return
    return str(x) + '_' + first_word

# List of FAO area codes for countries in sub-saharan africa and for regions
SSAareas = [str(i) for i in [7,53,20,233,29,35,32,37,39,45,46,107,250,72,61,178,209,238,74,75,81,90,175,114,
122,123,129,130,133,136,137,144,147,158,159,184,193,195,196,197,201,202,277,217,
226,215,251,181]]

regions = [str(i) for i in [420,3455,5000,5100,5103,5200,5203,5205,5300,5400,5500]]

# GDP (US$ per capita, 2015 prices) and Gross Fixed Capital Formation (Share of GDP) for each region (except SSA)
MK_r = (faostat.get_data_df('MK', pars = {'areas' : [regions], 
  'elements': [6185,6187], 'items' : [22008,22015]}).pivot(index=['Area','Year'], columns='Item', values='Value')
  )


# Gross Fixed Capital Formation (Agriculture, Forestry and Fishing)- Share of Gross Fixed Capital Formation US$, 2015 prices
CS = faostat.get_data_df('CS', pars = {'areas' : SSAareas, 'elements': 61392}).pivot(index=['Area','Year'], columns='Item', values='Value')

# Calculate population for each country in SSA, for per-capita weighting
MK_pop = faostat.get_data_df('MK', pars = {'areas' : SSAareas, 'elements': [6179,6185,6187], 'items' : [22008,22015]}).pivot(index=['Area','Year'], columns=['Item','Element'], values=['Value'])
MK_pop.columns = MK_pop.columns.map('_'.join).str.strip('_')
MK_pop['pop'] = MK_pop['Value_Gross Domestic Product_Value US$, 2015 prices'] / MK_pop['Value_Gross Domestic Product_Value US$ per capita, 2015 prices']

# Add Gross Fixed Capital Formation (Agriculture, Forestry and Fishing) to population dataframe
MK_pop = MK_pop.join(CS)
MK_pop = MK_pop.reset_index()

# Calculate GDP per capita average for SSA region (values in each country weighted by population)
MK_SSAg = (MK_pop.groupby('Year').apply(weighted_average, 'Value_Gross Domestic Product_Value US$ per capita, 2015 prices', 'pop')
  .to_frame().reset_index().rename(columns = {0:'Gross Domestic Product'}).set_index('Year')
  )

# Calculate Capital Formation Share of GDP for SSA region (values in each country weighted by country GDP per capita) 
MK_SSAf = (MK_pop.groupby('Year').apply(weighted_average, 'Value_Gross Fixed Capital Formation_Share of GDP US$, 2015 prices', 'Value_Gross Domestic Product_Value US$, 2015 prices')
  .to_frame().reset_index().rename(columns = {0:'Gross Fixed Capital Formation'}).set_index('Year')
  )

# Calculate Share of Capital Formation going to Ag Forest Fish for SSA (values in each country weighted by country Capital Formation total) 
MK_CS = (MK_pop.groupby('Year').apply(weighted_average, 'Gross Fixed Capital Formation (Agriculture, Forestry and Fishing)', 'Value_Gross Fixed Capital Formation_Value US$, 2015 prices')
  .to_frame().reset_index().rename(columns = {0:'Gross Fixed Capital Formation (Agriculture, Forestry and Fishing)'}).set_index('Year').replace(0, np.nan)
  )

# Join SSA-aggregated variables to other regions
MK_SSAg = MK_SSAg.join(MK_SSAf)
MK_SSAg = pd.concat({'SSA': MK_SSAg}, names=['Area'])
MK_r = pd.concat([MK_r,MK_SSAg])

MK_CS = pd.concat({'SSA': MK_CS}, names=['Area'])
CS_r = faostat.get_data_df('CS', pars = {'areas' : [regions], 'elements': 61392}).pivot(index=['Area','Year'], columns='Item', values='Value')
CS_r = pd.concat([CS_r,MK_CS])

# Re-download same variables (GDP, Capital formation) for each SSA country (only the ones we actually need)
MK_SSA = faostat.get_data_df('MK', pars = {'areas' : SSAareas, 'elements': [6185,6187], 'items' : [22008,22015]}).pivot(index=['Area','Year'], columns='Item', values='Value')
CS_SSA = faostat.get_data_df('CS', pars = {'areas' : SSAareas, 'elements': 61392}).pivot(index=['Area','Year'], columns='Item', values='Value')


### leftover code from testing aggregation method

# MK_NA1 = MK_r.reset_index()[MK_r.reset_index().Area == 'Northern Africa'].iloc[:,1:4]
# MK_pop = faostat.get_data_df('MK', pars = {'areas' : '5103>', 'elements': [6179,6185,6187], 'items' : [22008,22015]})
# MK_pop = MK_pop.pivot(index=['Area','Year'], columns=['Item','Element'], values=['Value'])
# MK_pop['pop'] = MK_pop.iloc[:,1] / MK_pop.iloc[:,0]
# MK_pop = MK_pop.reset_index()
# MK_pop.columns = MK_pop.columns.map('_'.join).str.strip('_')
# MK_NA2 = MK_pop.groupby('Year').apply(weighted_average, 'Value_Gross Domestic Product_Value US$ per capita, 2015 prices', 'pop').to_frame().reset_index()
# MK_NA3 = MK_pop.groupby('Year').apply(weighted_average, 'Value_Gross Fixed Capital Formation_Share of GDP US$, 2015 prices', 'Value_Gross Domestic Product_Value US$, 2015 prices').to_frame().reset_index()
# MK_NA = MK_NA1.merge(MK_NA3)
# MK_NA['diff'] = MK_NA.iloc[:,3] - MK_NA.iloc[:,2]

del MK_pop, MK_SSAg, MK_SSAf, CS, MK_CS


# Download and reshape Emissions variables (kilotonnes)
GT = pd.read_csv('../raw data/FAOSTAT_GT.csv').pivot(index=['Area Code (FAO)','Area','Year'], columns='Element', values='Value').reset_index()
GT['Year'] = GT['Year'].astype(str)
GT_r = GT[GT['Area Code (FAO)'].astype(str).isin(regions)].drop('Area Code (FAO)', axis=1).set_index(['Area','Year'])
GT_SSA = GT[GT['Area Code (FAO)'].astype(str).isin(SSAareas)].drop('Area Code (FAO)', axis=1).set_index(['Area','Year'])

# Sum up emissions across all SSA countries to get emissions for region
GT_SSAg = pd.concat({'SSA': GT_SSA.reset_index().drop('Area', axis=1).groupby('Year').agg(sum)}, names=['Area'])
GT_r = pd.concat([GT_r,GT_SSAg])

# GT_NA = pd.read_csv('../raw data/FAOSTAT_GTNA.csv').pivot(index=['Area Code (FAO)','Area','Year'], columns='Element', values='Value')
# GT_NA1 = GT_NA.reset_index()[GT_NA.reset_index()['Area Code (FAO)'].astype(str).isin(regions)].drop('Area Code (FAO)', axis=1).set_index(['Area','Year'])
# GT_NA2 = GT_NA.reset_index()[~(GT_NA.reset_index()['Area Code (FAO)'].astype(str).isin(regions))].drop('Area Code (FAO)', axis=1).set_index(['Area','Year'])
# 
# GT_NA3 = pd.concat({'Northern Africa': GT_NA2.reset_index().drop('Area', axis=1).groupby('Year').agg(sum)}, names=['Area'])
# GT_NA = pd.concat([GT_NA1,GT_NA3], axis=1)
# 
# GT_NA['diff'] = GT_NA.iloc[:,1] - GT_NA.iloc[:,0]


# Download and reshape Production variables (tonnes)
QCL = pd.read_csv('../raw data/FAOSTAT_QCL.csv').pivot(index=['Area Code (FAO)','Area','Year'], columns='Item', values='Value')
QCL = QCL.rename(columns={col: transform_colname(col,'prod') for col in QCL.columns}).reset_index()
QCL['Year'] = QCL['Year'].astype(str)
QCL_r = QCL[QCL['Area Code (FAO)'].astype(str).isin(regions)].drop('Area Code (FAO)', axis=1).set_index(['Area','Year'])
QCL_SSA = QCL[QCL['Area Code (FAO)'].astype(str).isin(SSAareas)].drop('Area Code (FAO)', axis=1).set_index(['Area','Year'])

QCL_SSAg = pd.concat({'SSA': QCL_SSA.reset_index().drop('Area', axis=1).groupby('Year').agg(sum)}, names=['Area'])
QCL_r = pd.concat([QCL_r,QCL_SSAg]).sort_index()

# Download and reshape Trade variables (1000 US$)
TCL = pd.read_csv('../raw data/FAOSTAT_TCL.csv').pivot(index=['Area Code (FAO)','Area','Year'], columns=['Element','Item'], values='Value').reset_index()
TCL.columns = TCL.columns.map('_'.join).str.strip('_')
TCL['Year'] = TCL['Year'].astype(str)
TCL_r = TCL[TCL['Area Code (FAO)'].astype(str).isin(regions)].drop('Area Code (FAO)', axis=1).set_index(['Area','Year'])
TCL_SSA = TCL[TCL['Area Code (FAO)'].astype(str).isin(SSAareas)].drop('Area Code (FAO)', axis=1).set_index(['Area','Year'])

TCL_SSAg = pd.concat({'SSA': TCL_SSA.reset_index().drop('Area', axis=1).groupby('Year').agg(sum)}, names=['Area'])
TCL_r = pd.concat([TCL_r,TCL_SSAg])

#Download and reshape Foreign Direct Investment (US$ million, 2015)
FDI_r = faostat.get_data_df('FDI', pars = {'areas' : regions, 'elements' : [6179], 'items' : [23082, 23085]}).pivot(index=['Area','Year'], columns='Item', values='Value')
FDI_SSA = faostat.get_data_df('FDI', pars = {'areas' : SSAareas, 'elements' : [6179], 'items' : [23082, 23085]}).pivot(index=['Area','Year'], columns='Item', values='Value')

FDI_SSAg = pd.concat({'SSA': FDI_SSA.reset_index().drop('Area', axis=1).groupby('Year').agg(sum)}, names=['Area'])
FDI_r = pd.concat([FDI_r,FDI_SSAg])


del GT, GT_SSAg, QCL, QCL_SSAg, FDI_SSAg, TCL, TCL_SSAg


# Food Security variables from FAOSTAT
FS = faostat.get_data_df('FS', pars = {'areas' : [regions,SSAareas], 'elements': 6120}).pivot(index=['Area Code (FAO)','Area','Year'], columns='Item', values='Value').fillna(value=np.nan)
FS = FS.rename(columns={col: 'FS_' + col for col in FS.columns})

# Recode the year of three-year-average values to the middle year
FS1 = FS.reset_index()[FS.reset_index().Year.apply(len) > 4].dropna(how='all', axis=1)
FS1['Year'] = FS1['Year'].apply(lambda x: str(int(x[0:4])+1))
FS1 = FS1.set_index(['Area Code (FAO)','Area','Year'])
FS2 = FS.reset_index()[~(FS.reset_index().Year.apply(len) > 4)].dropna(how='all', axis=1).set_index(['Area Code (FAO)','Area','Year'])
FS = FS1.join(FS2).reset_index()
FS['Area'] = FS['Area'].replace('Sub-Saharan Africa', 'SSA')

# Separate region-level data from country-level data
FS_r = FS[FS['Area Code (FAO)'].isin(regions)].drop('Area Code (FAO)', axis=1).set_index(['Area','Year']).sort_index()
FS_SSA = FS[FS['Area Code (FAO)'].isin(SSAareas)].drop('Area Code (FAO)', axis=1).set_index(['Area','Year']).sort_index()

# Add Land Area and generate average rail density for SSA
SSA_size = faostat.get_data_df('RL', pars = {'areas' : [SSAareas], 'elements': 5110, 'items': 6601}).pivot(index=['Area Code (FAO)','Area','Year'], columns='Item', values='Value').fillna(value=np.nan)
FS_SSA = FS_SSA.join(SSA_size).reset_index()
FS_SSA['FS_Rail lines density (total route in km per 100 square km of land area)'] = FS_SSA['FS_Rail lines density (total route in km per 100 square km of land area)'].astype(np.float64)
FS_SSA = FS_SSA.set_index(['Area','Year'])
FS_SSAg = (FS_SSA.groupby('Year').apply(weighted_average, 'FS_Rail lines density (total route in km per 100 square km of land area)', 'Land area')
  .to_frame().reset_index().rename(columns = {0:'FS_Rail lines density (total route in km per 100 square km of land area)'}).set_index('Year')
  .replace(0, np.nan)
  )

FS_SSAg = pd.concat({'SSA': FS_SSAg}, names=['Area'])
  

FS_r['FS_Rail lines density (total route in km per 100 square km of land area)'] = (
  pd.concat([pd.DataFrame(FS_r.reset_index()[FS_r.reset_index().Area != 'SSA']
  .set_index(['Area', 'Year'])
  ['FS_Rail lines density (total route in km per 100 square km of land area)']),FS_SSAg])
  .sort_index()
  )

del FS1, FS2, FS


FAO_r = MK_r.join([CS_r, FDI_r, GT_r, QCL_r, TCL_r, FS_r]).sort_index()
FAO_SSA = MK_SSA.join([CS_SSA, FDI_SSA, GT_SSA, QCL_SSA, TCL_SSA, FS_SSA]).sort_index()

# View(which_years(FAO_r))
# View(which_years(FAO_SSA))

FAO_r.reset_index().to_csv('../data/FAO_r.csv', index=False)
FAO_SSA.reset_index().to_csv('../data/FAO_SSA.csv', index=False)

# dfs = dict(MK_r=MK_r, CS_r=CS_r, FDI_r=FDI_r, GT_r=GT_r, QCL_r=QCL_r, TCL_r=TCL_r, FS_r=FS_r)
# 
# for name, df in dfs.items():
#   print(name)
#   df.sort_index().reset_index().Area.drop_duplicates()
#   df.sort_index().reset_index().Area.drop_duplicates().count()

# print(pd.DataFrame(faostat.get_areas('QCL').items()).sort_values(0).to_string())
