from itertools import product
import numpy as np
import pandas as pd
from scipy.stats import describe

##### inspired by meta-features in
# Two-Stage Transfer Surrogate Model for Automatic Hyperparameter Optimization
# by Martin Wistuba, Nicolas Schilling, and Lars Schmidt-Thieme
# https://github.com/wistuba/TST

def extract(data, targets):
    
    oh_bal = []
    descriptions = []
    for ft in range(data.shape[1]):
        if np.unique(data[:,ft]).size == 2:
            oh_bal.append(np.count_nonzero(data[:,ft]) / data.shape[0])
        else:
            try:
                local = data[:,ft].todense() # sparse not supported by describe
            except Exception:
                local = data[:,ft]
            descriptions.append(describe(local)._asdict())

    _, freqs = np.unique(targets, return_counts=True)
    results = {
        'n_instances': data.shape[0],
        'n_predictors': data.shape[1],
        'n_classes': freqs.size,
        'class_freq_mean': np.mean(freqs),
        'class_freq_std': np.std(freqs),
        'n_num': data.shape[1] - len(oh_bal),
        'n_onehot': len(oh_bal),
        'onehot_bal_mean': np.mean(oh_bal),
        'onehot_bal_std': np.std(oh_bal)
    }

    if len(descriptions) > 0:
        descriptions = pd.DataFrame(descriptions)
        descriptions[['min', 'max']] = pd.DataFrame(descriptions['minmax'].tolist(), index=descriptions.index)
        descriptions = descriptions.drop(['nobs', 'minmax'], axis=1)
        
        for col, agg in product(descriptions.columns, [np.mean, np.std]): #, np.min, np.max]):
            results[f'{col}_{agg.__name__}'] = agg(descriptions[col].astype(float))

    return pd.Series(results)