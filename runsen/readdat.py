import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# pip install scikit-learn
# pip install openpyxl
from sklearn.decomposition import PCA
from numpy.linalg import inv
from statsmodels.stats.outliers_influence import variance_inflation_factor
from names import bbg2name # colm change 0
## 2. Load and Preprocess Data from 2004-2024; from six Russell and Bloomberg Barclays bond indices
index_data = pd.read_excel("data/org_data.xlsx",sheet_name = "Sheet2")
index_data.drop('TX60AR Index',axis=1,inplace=True) # colm change 1
# clean up blanks
fixblanks={x:x.replace("\u202f"," ") for x in list(index_data.columns)}
index_data=index_data.rename(columns=fixblanks).rename(columns=bbg2name) # colm 1
list(bbg2name[x] if x in bbg2name.keys() else 'not found' for x in list(index_data.columns))
index_data.columns.sort_values()
sorted(list(bbg2name.keys()))
set(index_data.columns)-set(bbg2name.keys())
index_data['Date'] = pd.to_datetime(index_data['Date'])
index_data.set_index('Date', inplace=True)
index_returns = index_data.fillna(method='ffill').pct_change().dropna() # colm change 2
returns_clean = index_returns.dropna()
index_returns.to_csv('data/org_data_copy.csv') # colm 3

# Save index_returns to a pickle file
index_returns.to_pickle('data/index_returns.pkl') # Save to pickle


# Test the collinearity

vif_data = pd.DataFrame()
vif_data["Asset"] = returns_clean.columns
vif_data["VIF"] = [variance_inflation_factor(returns_clean.values, i) for i in range(returns_clean.shape[1])]

print(vif_data)