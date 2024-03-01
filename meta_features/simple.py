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
    n_num = data.shape[1] - len(oh_bal)
    results = {
        'n_instances': data.shape[0],
        'n_predictors': data.shape[1],
        'n_classes': freqs.size,
        'class_freq_mean': np.mean(freqs),
        'class_freq_std': np.std(freqs),
        'n_num': n_num,
        'n_onehot': len(oh_bal),
        'num_onehot_ratio': (n_num / len(oh_bal)) / data.shape[1] if len(oh_bal) > 0 else 1,
        'onehot_bal_mean': np.mean(oh_bal) if len(oh_bal) > 0 else 0,
        'onehot_bal_std': np.std(oh_bal) if len(oh_bal) > 0 else 0,
        'onehot_bal_max': np.max(oh_bal) if len(oh_bal) > 0 else 0,
        'onehot_bal_min': np.min(oh_bal) if len(oh_bal) > 0 else 0
    }

    if len(descriptions) > 0:
        descriptions = pd.DataFrame(descriptions)
        descriptions[['min', 'max']] = pd.DataFrame(descriptions['minmax'].tolist(), index=descriptions.index)
        descriptions = descriptions.drop(['nobs', 'minmax'], axis=1)
        
        for col, agg in product(descriptions.columns, [np.mean, np.std]): #, np.min, np.max]):
            agg_val = agg(descriptions[col].astype(float))
            results[f'{col}_{agg.__name__}'] = 0 if np.isnan(agg_val) else agg_val

    return pd.Series(results)