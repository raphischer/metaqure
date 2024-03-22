import warnings

import argparse
import os

from tqdm import tqdm
import pandas as pd
import numpy as np

from data_loading import ALL_DS, data_variant_loaders

warnings.filterwarnings('ignore')

META_FT_DIR = 'meta_features'

try:
    from meta_features.ds2vec import extract as ds2vec
    from meta_features.statistical import extract as statistical
    from meta_features.pca import extract as pca
    EXTRACTORS = [statistical, pca, ds2vec]
except ImportError:
    print('Could not load libraries for calculating new meta-features, but the ones in the repository can be loaded')


def load_meta_features():
    meta_features = {}
    for meta_ft_file in os.listdir(META_FT_DIR):
        if not '.csv' in meta_ft_file:
            continue
        meta_features[meta_ft_file.replace('.csv', '')] = pd.read_csv(os.path.join(META_FT_DIR, meta_ft_file)).set_index('Unnamed: 0').fillna(0)
    meta_features['combined'] = pd.concat(meta_features.values(), axis=1)
    return meta_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data and model input
    parser.add_argument("--data-home", default="/data/d1/sus-meta-results/data")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use")
    args = parser.parse_args()

    for extractor in EXTRACTORS:
        all_metafeatures, ds_index = [], []
        for ds, subsample in tqdm(ALL_DS):
            variant_loaders = data_variant_loaders(ds, args.data_home, args.seed, subsample)
            for ds_variant in variant_loaders:
                X_train, X_test, y_train, y_test, _, ds_name = ds_variant()
                X, y = np.concatenate([X_train, X_test]), np.concatenate([y_train, y_test])
                all_metafeatures.append( extractor(X, y) )
                ds_index.append(ds_name)
        ft_file = os.path.join(META_FT_DIR, f"{extractor.__module__.split('.')[-1]}.csv")
        pd.DataFrame(all_metafeatures, index=ds_index).to_csv(ft_file)
