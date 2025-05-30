
# Sensitivity Decomposition Project

This repository organizes all code, data references, and analysis results related to sensitivity-based portfolio decomposition.

## 🔧 Project Structure
```text
my-python-project/
│
├── src/                       # Core logic scripts
│   ├── readdat.py             # Data loading and VIF calculation
│   └── runsens.py             # Shrinkage and PCA sensitivity simulation
│
├── data/                      # Raw and preprocessed data (not committed to GitHub)
│   ├── org_data1.xlsx
│   └── index_returns.pkl
│
├── results/                   # Output results (for final inspection)
│   ├── result_PCA.csv
│   └── result_diff_shrinkage_rates.csv
│
├── notebooks/
│   └── test_runsens.ipynb     # Result verification and plotting
│
├── README.md                  # Project overview (this file)
└── requirements.txt           # Python dependencies
```


