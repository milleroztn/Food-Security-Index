# Function to calculate weighted average
def weighted_average(dataframe, value, weight):
    val = dataframe[value]
    wt = dataframe[weight]
    return (val * wt).sum() / wt.sum()

# Function to summarize which variables are available for which years in which country
def which_years(x):

    rows = []
    
    for region in x.reset_index()['Area'].unique():
        for variable in x.columns:
            years_not_missing = []
            for year in x.reset_index().loc[(x.reset_index()['Area'] == region), 'Year']:
                value = x.reset_index().loc[(x.reset_index()['Area'] == region) & (x.reset_index()['Year'] == year), variable].values[0]
                if pd.notna(value):
                    years_not_missing.append(year)
                    
            if len(years_not_missing) > 0:
                min_year = min(years_not_missing)
                max_year = max(years_not_missing)
                num_missing = (int(max_year) - int(min_year) + 1) - len(years_not_missing)
            else:
                min_year = 'None'
                max_year = 'None'
                num_missing = 'None'
                    
            row = {
                'Region': region,
                'Variable': variable,
                'Min Year': min_year,
                'Max Year': max_year,
                'Num Years Missing': num_missing
            }
            rows.append(row)
            
    return pd.DataFrame(rows)
  
# Function to transform column names, add 'prod' or whatever before first word
def transform_colname(name,mod):
    first_word = name.split()[0]
    if first_word[-1] == ',':
        first_word = first_word[:-1]
    return str(mod) + '_' + first_word
  
  
# Function to compare areas in dataset
def which_areas(df1,df2):
  print("Values in df1 that are not in df2:")
  s1 = df1.reset_index().Area[~df1.reset_index().Area.isin(df2.reset_index().Area)].drop_duplicates()
  print(s1)
  print("Values in df2 that are not in df1:")
  s2 = df2.reset_index().Area[~df2.reset_index().Area.isin(df1.reset_index().Area)].drop_duplicates()
  print(s2)
