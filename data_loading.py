import argparse

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import pandas as pd

from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from scipy import sparse

# Sklearn real-life datasets
SKLEARN_DATASETS = [
    'olivetti_faces',
    'lfw_people',
    'lfw_pairs',
    '20newsgroups_vectorized',
    'covtype',
    # 'kddcup99', # TODO fix error
    # 'california_housing', # TODO fix error
    # too small / not stored locally:
    # 'breast_cancer',
    # 'digits',
    # 'iris',
    # 'wine'
]

# Popular OpenML datasets
OPENML_DATASETS = [
    # verified multiclass sorted by number of runs
    'credit-g',
    'blood-transfusion-service-center',
    'monks-problems-2',
    'tic-tac-toe',
    'monks-problems-1',
    'steel-plates-fault',
    'kr-vs-kp',
    'qsar-biodeg',
    'wdbc',
    'phoneme',
    'diabetes',
    'ozone-level-8hr',
    'hill-valley',
    'kc2',
    'eeg-eye-state',
    'climate-model-simulation-crashes',
    'spambase',
    'kc1',
    'ilpd',
    'pc1',
    'pc3',

    # additional verified multiclass sorted by number of likes (at least three)
    'SpeedDating',
    'mnist_784',
    'banknote-authentication',
    'adult',
    'Titanic',
    'Satellite',
    'bank-marketing',

    # additional verified multiclass sorted by number of downloads (at least 40)
    'one-hundred-plants-texture',
    'arrhythmia',
    'amazon-commerce-reviews',
    'one-hundred-plants-shape',
    'Bioresponse',
]

DATASETS = SKLEARN_DATASETS + OPENML_DATASETS


def load_sklearn_feature_names(ds):
    if hasattr(ds, 'feature_names'):
        return ds.feature_names
    else:
        return [f'feat_{idx}' for idx in range(ds.data.shape[1])]
    

def load_sklearn(ds_name, data_home=None):
    ds_loader = getattr(datasets, f'fetch_{ds_name}') if hasattr(datasets, f'fetch_{ds_name}') else getattr(datasets, f'load_{ds_name}')
    try:
        # some datasets come with prepared split
        ds_train = ds_loader(subset='train', data_home=data_home)
        X_train = ds_train.data
        y_train = ds_train.target
        ds_test = ds_loader(subset='test', data_home=data_home)
        X_test = ds_test.data
        y_test = ds_test.target
        feature_names = load_sklearn_feature_names(ds_train)
        if X_train.shape == X_test.shape:
            raise TypeError # some data sets allow for specific subsets, but return the full dataset if subset is not selected well
    except TypeError:
        ds = ds_loader(data_home=data_home)
        feature_names = load_sklearn_feature_names(ds)
        X_train, X_test, y_train, y_test = train_test_split(ds.data, ds.target)

    return X_train, X_test, y_train, y_test, feature_names


def load_openml(ds_name, data_home=None):
    data = datasets.fetch_openml(name=ds_name, data_home=data_home, parser='auto')
    X = pd.get_dummies(data['data']).astype(float) # one-hot
    X, feature_names = X.values, X.columns.values
    y, cat = pd.factorize(data['target'])
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test, feature_names


def load_data(ds_name, data_home=None):
    if ds_name in SKLEARN_DATASETS:
        X_train, X_test, y_train, y_test, feature_names = load_sklearn(ds_name, data_home)
    elif ds_name in OPENML_DATASETS:
        X_train, X_test, y_train, y_test, feature_names = load_openml(ds_name, data_home)
    else:
        raise RuntimeError(f'Dataset {ds_name} not found!')
    
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    X_train = imp.fit_transform(X_train)
    X_test = imp.fit_transform(X_test)
    return X_train, X_test, y_train, y_test, feature_names




# def label_encoding(X_train, X_test=None):
#     old_shape = X_train.shape
#     if X_test is None:
#         data = X_train
#     else:
#         data = np.concatenate([X_train, X_test])
#     categorical = []
#     if len(data.shape) > 1:
#         for column in range(data.shape[1]):
#             try: 
#                 float_col = data[:, column].astype(float)
#                 data[:, column] = float_col
#             except Exception:
#                 categorical.append(column)
#                 data[:, column] = preprocessing.LabelEncoder().fit_transform(data[:, column])
#     else:
#         data = preprocessing.LabelEncoder().fit_transform(data)
#     if X_test is None:
#         X_train = data
#     else:
#         X_train, X_test = np.split(data, [X_train.shape[0]])
#     assert(X_train.shape == old_shape)
#     return X_train, X_test, categorical
    

    # remove labels & rows that are only present in one split
    # train_labels = set(list(y_train))
    # test_labels = set(list(y_test))
    # for label in train_labels:
    #     if label not in test_labels:
    #         where = np.where(y_train != label)[0]
    #         X_train, y_train = X_train[where], y_train[where]
    # for label in test_labels:
    #     if label not in train_labels:
    #         where = np.where(y_test != label)[0]
    #         X_test, y_test = X_test[where], y_test[where]
    # # use label encoding for categorical features and labels
    # try:
    #     X_train = X_train.astype(float)
    #     X_test = X_test.astype(float)
    #     categorical_columns = []
    # except ValueError:
    #     X_train, X_test, categorical_columns = label_encoding(X_train, X_test)
    # try:
    #     y_train = y_train.astype(int)
    #     y_test = y_test.astype(int)
    # except ValueError:
    #     y_train, y_test, _ = label_encoding(y_train, y_test)
    # # impute nan values
    # imp = SimpleImputer(missing_values=np.nan, strategy='median')
    # X_train = imp.fit_transform(X_train)
    # X_test = imp.fit_transform(X_test)
    # # identify the unique categorical values of each column
    # cat_vals = [np.array(sorted(set(np.concatenate([np.unique(X_train[:, col]), np.unique(X_test[:, col])])))) for col in categorical_columns]
    # # onehot encoding for categorical features, standard-scale all non-categoricals
    # if not sparse.issparse(X_train):
    #     scaler = ColumnTransformer([
    #         ('categorical', preprocessing.OneHotEncoder(categories=cat_vals), categorical_columns)
    #     ], remainder=preprocessing.StandardScaler())
    #     X_train = scaler.fit_transform(X_train)
    #     X_test = scaler.transform(X_test)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-home", default=None)

    args = parser.parse_args()

    size_ds = []
    for ds in DATASETS:
        X_train, X_test, y_train, y_test, feat = load_data(ds, args.data_home)
        tr_s, te_s, n_class = y_train.size,  y_test.size, np.unique(y_test).size
        print(f'{ds[:20]:<20} {tr_s + te_s:>6} ({tr_s / (tr_s + te_s) * 100:4.1f}% train) instances  {n_class:>4} classes  {len(feat):>7} feat - {str(feat)[:50]} ...')
        size_ds.append( (tr_s + te_s, ds) )

    print('Ordered by size:')
    print(' '.join([ f'"{ds}"' for _, ds in sorted(size_ds) ]))
