from itertools import product

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def extract(data, targets, n_comp=3):
    try:
        data = np.asarray(data.todense())
    except AttributeError:
        pass
    concat = np.concatenate([data, targets[:,None]], axis=1)
    pca = make_pipeline(StandardScaler(), PCA(n_components=n_comp))
    reduced = pca.fit_transform(concat)

    return {f'{idx}_{agg.__name__}': agg(reduced[idx]) for idx, agg in product(np.arange(n_comp), [np.min, np.max, np.mean, np.std, np.median])}
