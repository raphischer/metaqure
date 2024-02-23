import argparse
import os

from tqdm import tqdm
import pandas as pd

from data_loading import DATASETS, load_data
from meta_features.ds2vec import extract as ds2vec
from meta_features.simple import extract as simple
from meta_features.pca import extract as pca

EXTRACTORS = [pca, simple, ds2vec]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data and model input
    parser.add_argument("--data-home", default="/data/d1/sus-meta-results/data")
    args = parser.parse_args()

    for extractor in EXTRACTORS:
        all_metafeatures = []
        for ds in tqdm(DATASETS):
            X_train, _, y_train, _, _ = load_data(ds, args.data_home)
            all_metafeatures.append( extractor(X_train, y_train) )
        ft_file = os.path.join("meta_features", f"{extractor.__module__.split('.')[-1]}.csv")
        pd.DataFrame(all_metafeatures, index=DATASETS).to_csv(ft_file)
