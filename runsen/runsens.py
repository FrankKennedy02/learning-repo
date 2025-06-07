# Hi Yanni -- I made this change and committed it.  I'll ask you to confirm you see the change.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# pip install scikit-learn
# pip install openpyxl
from sklearn.decomposition import PCA
from numpy.linalg import inv
from statsmodels.stats.outliers_influence import variance_inflation_factor
from names import bbg2name # colm change 0

def simulate_sensitivity_model(index_returns, shrinkage_lambda=0,annualization_factor = 12 , gamma=0.01, n_sim=1000):
    T = 100 # index_returns.shape[0]
    n_assets = index_returns.shape[1]

    # Estimated covariance (annualized)
    Sigma = index_returns.cov().values * annualization_factor 
    vol = np.sqrt(np.diag(Sigma))
    vol_df = pd.DataFrame({
        'Asset': index_returns.columns,
        'Volatility': vol
    })
    
    # print("Vol", vol_df)
    # Apply shrinkage if needed
    if shrinkage_lambda > 0:
        # shrink_target = np.identity(n_assets) * np.trace(Sigma) / n_assets
        shrink_target = np.diag(np.diag(Sigma))
        Sigma = (1 - shrinkage_lambda) * Sigma + shrinkage_lambda * shrink_target


    # Omega_0 = covariance of theta
    Omega_0 = Sigma / T
    theta_0 = np.zeros((n_assets, 1))  # mean zero
    return MLBsim(Sigma,Omega_0,theta_0,gamma,n_sim)

def MLBsim(Sigma,Omega_0,theta_0,gamma,n_sim):
    # M² = trace(Sigma⁻¹) / n
    n_assets = Sigma.shape[0]
    Sigma_inv = np.linalg.inv(Sigma)
    trace_Sigma_inv = np.trace(Sigma_inv)
    M = np.sqrt(trace_Sigma_inv / n_assets)

    # L = 1 because Γ = Omega is fixed
    L = 1.0

    # Simulate theta and compute h
    h_1_norm_list = []
    h_2_norm_list = []

    for _ in range(n_sim):
        theta = np.random.multivariate_normal(mean=np.zeros(n_assets), cov=Omega_0).reshape(-1, 1)
        h = gamma * Sigma_inv @ theta
        h_1 = np.sum(np.abs(h))
        h_2 = np.sum(h ** 2)
        h_1_norm_list.append(h_1)
        h_2_norm_list.append(h_2)

    # Compute expectation estimates
    E_l1_squared = (np.mean(h_1_norm_list)) ** 2
    E_l2 = np.mean(h_2_norm_list)

    B = E_l1_squared / E_l2
    S = M * L * np.sqrt(B)

    return { "M": M, "L": L,"B": B, "S": S}

def simulate_sensitivity_pca_model(index_returns,annualization_factor=12, n_factors=None, gamma=0.01, n_sim=1000):
    n_returns, n_assets = index_returns.shape
    T=100
    returns_matrix = index_returns.values
    len(np.argwhere(np.isnan(returns_matrix)))
    Sigma = np.cov(returns_matrix.T) * annualization_factor 
    theta_0 = np.zeros((n_assets, 1))  # mean zero
    # === Step 1: PCA Decomposition ===
    n_factors = min(n_factors, T, n_assets)
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    evecn=eigenvectors[:,-n_factors:]
    evaln=eigenvalues[-n_factors:]
    Sigma_pca=evecn@np.diag(evaln)@evecn.T
    np.fill_diagonal(Sigma_pca, np.diag(Sigma))
    explained_variance_ratio= np.sum(evaln) / np.sum(eigenvalues)  # Explained variance ratio
    Omega_0= Sigma_pca / T
    mlb= MLBsim(Sigma_pca,Omega_0,theta_0,gamma,n_sim)
     
    mlb.update({"explained_variance_ratio":explained_variance_ratio, "n_factors": n_factors})
    return mlb
    

def run_sensitivity_analysis(index_returns, gamma, shrinkage_vec,n_factors_vec,time_windows):
    results_fixed = []
    results_random = []
    for window in time_windows:
        end_date = index_returns.index.max()
        start_date = end_date - pd.DateOffset(years=window)
        data_slice = index_returns[start_date:]
        for i in np.arange(len(shrinkage_vec)):
            result_random = simulate_sensitivity_model(data_slice,  gamma=gamma, shrinkage_lambda=shrinkage_vec[i])
            result_random.update({"Shrinkage Lambda": shrinkage_vec[i],"Time Window (Years)": window})
            results_random.append(result_random)
        for i in np.arange(len(n_factors_vec)):
            result_fixed = simulate_sensitivity_pca_model(data_slice,  n_factors=n_factors_vec[i],gamma=gamma)
            result_fixed.update({"Time Window (Years)": window})
            results_fixed.append(result_fixed)         
    results_df_fixed = pd.DataFrame(results_fixed)
    print(results_df_fixed)
    results_df_fixed.to_csv(f"sensitivity_summary_generalized_PCA_{100*gamma}%.csv", index=False)
    
    results_df_random = pd.DataFrame(results_random)
    results_df_random.to_csv(f"sensitivity_summary_generalized_random_{100*gamma}%.csv", index=False)
    return results_df_fixed,results_df_random
    
## 2. Load and Preprocess Data from 2004-2024; from six Russell and Bloomberg Barclays bond indices
# Reload index_returns from the pickle file
returns_clean = pd.read_pickle('data/index_returns.pkl') # Reload from pickle
shrinkage_vec = [0, 0.2, 0.4, 0.6, 0.8]
n_factors_vec=np.arange(0,24,1)

time_windows = [10, 20]
gamma = 0.01
annualization_factor=12

# Run and display
r0,r1 = run_sensitivity_analysis(returns_clean, gamma, shrinkage_vec,n_factors_vec, time_windows)

print(r0)
print(r1)