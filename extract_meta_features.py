import warnings

import argparse
import os

from tqdm import tqdm
import pandas as pd
import numpy as np

from data_loading import ALL_DS, data_variant_loaders
from meta_features.ds2vec import extract as ds2vec
from meta_features.simple import extract as simple
from meta_features.pca import extract as pca

warnings.filterwarnings('ignore')

EXTRACTORS = [simple, pca, ds2vec]

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
        ft_file = os.path.join("meta_features", f"{extractor.__module__.split('.')[-1]}.csv")
        pd.DataFrame(all_metafeatures, index=ds_index).to_csv(ft_file)
