def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import argparse
import os
from itertools import product

import pandas as pd
import numpy as np
from seedpy import fixedseed

from sklearn.dummy import DummyRegressor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, max_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor

from data_loading import ds_cv_split
from process_logs import PROPERTIES
from strep.index_and_rate import rate_database, load_database
from strep.util import load_meta, prop_dict_to_val


SCORING = {
    'MAE': mean_absolute_error,
    'MaxE': max_error
}
CV_SCORER = 'test_MAE'


ENCODERS = {
    'num': lambda config: Pipeline(steps=[ ('scaler', StandardScaler()) ]),
    'cat': lambda config: Pipeline(steps=[ ('onehot', OneHotEncoder(drop=config[0], categories=config[1])) ])
}


REGRESSORS = {
    'Global Mean':              (DummyRegressor, {}),
    'Linear Regression':        (LinearRegression, {}),
    'Ridge A1':                 (Ridge, {'alpha': 1.0}),
    'Ridge A0.1':               (Ridge, {'alpha': 0.1}),
    'Ridge A00.1':              (Ridge, {'alpha': 0.01}),
    'Lasso A1':                 (Lasso, {'alpha': 1.0}),
    'Lasso A0.1':               (Lasso, {'alpha': 0.1}),
    'Lasso A0.01':              (Lasso, {'alpha': 0.01}),
    'ElasticNet A1':            (ElasticNet, {'alpha': 1.0}),
    'ElasticNet A0.1':          (ElasticNet, {'alpha': 0.1}),
    'ElasticNet A0.01':         (ElasticNet, {'alpha': 0.01}),
    'LinearSVR C1':             (LinearSVR, {}),
    'LinearSVR C10':            (LinearSVR, {'C': 10.0}),
    'LinearSVR C100':           (LinearSVR, {'C': 100.0}),
    'SVR rbf':                  (SVR, {}),
    'SVR poly':                 (SVR, {'kernel': 'poly'}),
    'SVR sigmoid':              (SVR, {'kernel': 'sigmoid'}),
    'DecisionTree':             (DecisionTreeRegressor, {'max_depth': 5}),
    'FriedmanTree':             (DecisionTreeRegressor, {'max_depth': 5, 'criterion': 'friedman_mse'})
    # 'PoissonTree':              (DecisionTreeRegressor, {'max_depth': 5, 'criterion': 'poisson'})
}


def predict_with_all_models(X, y, regressors, cv_splitted, seed):
    pred_test = pd.DataFrame(index=X.index, columns=[regr for regr in regressors.keys()])
    pred_train = pd.DataFrame(index=X.index, columns=[f'{regr}_{split}' for regr, split in product(regressors.keys(), range(len(cv_splitted)))])
    for m_idx, (regr, (model_cls, params)) in enumerate(regressors.items()):
        if model_cls not in [DummyRegressor, LinearRegression, SVR]:
            params['random_state'] = seed
        clsf = model_cls(**params)
        # for models with intercept, onehot enocded features need to have one column dropped due to collinearity
        # https://stackoverflow.com/questions/44712521/very-large-values-predicted-for-linear-regression
        drop = 'first' if hasattr(clsf, 'fit_intercept') else None
        preprocessor = ColumnTransformer(transformers=[
            ('cat', Pipeline(steps=[ ('onehot', OneHotEncoder(drop=drop)) ]), ['model_enc']),
            ('num', Pipeline(steps=[ ('scaler', StandardScaler()) ]), X.drop('model_enc', axis=1).columns.tolist())
        ])
        
        # fit and predict for each split
        for split_idx, (train_idx, test_idx) in enumerate(cv_splitted):
            X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]
            model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', clsf)])
            model.fit(X_train, y_train)
            pred_train.iloc[train_idx,m_idx + split_idx] = model.predict(X_train)
            pred_test.iloc[test_idx,m_idx] = model.predict(X_test)
            # if hasattr(model, "predict_proba"):
        # merge the predictions of all train splits
        pred_train[regr] = pred_train[[c_ for c_ in pred_train.columns if regr in c_]].mean(axis=1)
        pred_train = pred_train.drop([f'{regr}_{split}' for split in range(len(cv_splitted))], axis=1)
    return pd.concat([pred_train, pred_test, pred_train - y.reshape(-1, 1), pred_test - y.reshape(-1, 1)], axis=1, keys=['train_pred', 'test_pred', 'train_err', 'test_err'])


def load_meta_features(dirname):
    meta_features = {}
    for meta_ft_file in os.listdir(dirname):
        if not '.csv' in meta_ft_file:
            continue
        meta_features[meta_ft_file.replace('.csv', '')] = pd.read_csv(os.path.join(dirname, meta_ft_file)).dropna().set_index('Unnamed: 0')
    meta_features['combined'] = pd.concat(meta_features.values(), axis=1)
    return meta_features


def error_info_as_string(row):
    return ' - '.join([f'{c.replace(f"{col}_", "")}: {row[c].abs().mean():7.3f} +- {row[c].abs().std():6.2f}' for c in row.columns if 'err' in c])


DB = 'exp_results/databases/ws3_240301.pkl'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-features-dir", default='meta_features')
    parser.add_argument("--seed", type=int, default=42, help="Seed to use")
    args = parser.parse_args()
    all_meta_features = load_meta_features(args.meta_features_dir)

    use_index = True

    with fixedseed(np, seed=args.seed):

        for ft_name, meta_features in all_meta_features.items():
            # load meta features and database        
            meta_ft_cols = list(meta_features.columns)
            database = load_database(DB)
            if use_index:
                meta = load_meta()
                rated_db = rate_database(database, meta, indexmode='best')[0]
                database = prop_dict_to_val(rated_db, 'index')
            # store meta feature in database
            for ds, sub_db in database.groupby('dataset'):
                if ds in meta_features.index:
                    database.loc[sub_db.index,meta_ft_cols] = meta_features[meta_features.index == ds].values
            database['model_enc'] = LabelEncoder().fit_transform(database['model'].values)
            meta_ft_cols.append('model_enc')

            # prepare grouped cross-validation
            meta_ds = set(database['dataset'].tolist())
            database = database.dropna()
            meta_ds = meta_ds - set(database['dataset'].tolist())
            X = database[meta_ft_cols]
            cv_splits = ds_cv_split(database['dataset'])
            results = pd.DataFrame(index=database.index)
            print(f'\n\n\n\n:::::::::::::::: META LEARN USING {ft_name} - SHAPE {X.shape} \nRemoved data sets:', meta_ds)

            for col in PROPERTIES['train'].keys():
                results = predict_with_all_models(X, database[col].values, REGRESSORS, cv_splits, args.seed)
                sorted_models = results['test_err'].abs().mean().sort_values()
                best_model_prediction = results.xs(sorted_models.index[0], level=1, axis=1)
                print(f'{ft_name:<8} - {col:<18} - Best Model: {sorted_models.index[0]:<17} - {error_info_as_string(best_model_prediction)}')
                for regr in sorted_models.index:
                    print(f'    {regr:<17} - {error_info_as_string(results.xs(regr, level=1, axis=1))})')
                results = pd.concat([results, best_model_prediction.rename(lambda c_ : f'{col}_{c_}', axis=1)], axis=1)
            results.to_pickle(os.path.join('exp_results', 'meta_learning', f'{ft_name}.pkl'))

            

            # # store true label and prediction in database
            # database[f'{col}_pred'] = predictions[best_name]
            # database[f'{col}_true'] = true[best_name]
            # database[f'{col}_prob'] = proba[best_name]
            # database[f'{col}_pred_model'] = best_name
            # database[f'{col}_pred_error'] = np.abs(database[f'{col}_pred'] - database[f'{col}_true'])
            # database['split_index'] = split_index
            
            # # write models in order to check feature importance later on
            # if col == COL_SEL:
            #     path = os.path.join('results', f'{COL_SEL}_models')
            #     if not os.path.isdir(path):
            #         os.makedirs(path)
            #     for idx, model in enumerate(best_models):
            #         with open(os.path.join(path, f'model{idx}.pkl'), 'wb') as outfile:
            #             pickle.dump(model, outfile)

            # # model = make_pipeline(StandardScaler(), LinearRegression())
            # # scores = cross_validate(model, X, y, return_train_score=True)

            # print_cv_scoring_results(meta_ft_file, SCORING.keys(), scores)

            # # res_str = ' - '.join([f'{key}: {np.mean(vals):7.2f} (+- {np.std(vals):7.2f})' for key, vals in cv.items()])
            # # print(f'{meta_name:<10}  :::  {res_str}')