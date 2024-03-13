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
from strep.index_and_rate import rate_database, load_database, index_to_value, calculate_single_compound_rating
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
        cat_cols = ['model_enc'] if 'environment_enc' not in X.columns else ['model_enc', 'environment_enc']
        preprocessor = ColumnTransformer(transformers=[
            ('cat', Pipeline(steps=[ ('onehot', OneHotEncoder(drop=drop)) ]), cat_cols),
            ('num', Pipeline(steps=[ ('scaler', StandardScaler()) ]), X.drop(cat_cols, axis=1).columns.tolist())
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
        meta_features[meta_ft_file.replace('.csv', '')] = pd.read_csv(os.path.join(dirname, meta_ft_file)).fillna(0).set_index('Unnamed: 0')
    meta_features['combined'] = pd.concat(meta_features.values(), axis=1)
    return meta_features


def error_info_as_string(row, col):
    return ' - '.join([f'{c.replace(f"{col}_", "")}: {row[c].abs().mean():10.6f} +- {row[c].abs().std():10.2f}' for c in row.columns if 'err' in c])


DB = 'exp_results/databases/complete.pkl'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-features-dir", default='meta_features')
    parser.add_argument("--seed", type=int, default=42, help="Seed to use")
    args = parser.parse_args()
    all_meta_features = load_meta_features(args.meta_features_dir)
    meta = load_meta()
    weights = {col: val['weight'] for col, val in meta['properties'].items()}

    with fixedseed(np, seed=args.seed):
        for ft_name, meta_features in all_meta_features.items():
            # load meta features and database        
            meta_ft_cols = list(meta_features.columns) + ['model_enc']
            database = load_database(DB)
            baselines = database[database['model'].isin(['PFN4', 'PFN16', 'PFN64'])]
            database = database.drop(baselines.index, axis=0)
            rated_db = rate_database(database, meta, indexmode='best')[0]
            index_db, value_db = prop_dict_to_val(rated_db, 'index'), prop_dict_to_val(rated_db, 'value')
            all_results, result_cols = [], []
            for db, scale in zip([index_db, value_db], ['index', 'value']):
                # store meta feature in database
                for ds, sub_db in db.groupby('dataset'):
                    if ds in meta_features.index:
                        db.loc[sub_db.index,meta_features.columns] = meta_features[meta_features.index == ds].values
                db['model_enc'] = LabelEncoder().fit_transform(db['model'].values)
                db['environment_enc'] = LabelEncoder().fit_transform(db['environment'].values)
                # prepare grouped cross-validation
                cv_splits = ds_cv_split(db['dataset'])
                for use_env, cols in zip(['not_use_env', 'use_env'], [meta_ft_cols, meta_ft_cols + ['environment_enc']]):
                    print(f'\n\n\n\n:::::::::::::::: META LEARN USING {ft_name} with {scale}, {use_env} \n')
                    compound_col_res_idc = {}
                    for col, col_meta in meta['properties'].items():
                        results = predict_with_all_models(db[cols], db[col].values, REGRESSORS, cv_splits, args.seed)
                        sorted_models = results['test_err'].abs().mean().sort_values()
                        best_model_prediction = results.xs(sorted_models.index[0], level=1, axis=1)
                        # for regr in sorted_models.index:
                        #     print(f'    {regr:<17} - {error_info_as_string(results.xs(regr, level=1, axis=1), col)})')
                        # TODO also store sorted_models.index[0] (best model name?)
                        all_results.append(best_model_prediction.rename(lambda c_ : f'{col}_{c_}', axis=1))
                        result_cols.append(f'{use_env}__{scale}')
                        compound_col_res_idc[col] = len(all_results) - 1 # remember the columns that will be important for the compound index
                        # recalculate index predictions to real value predictions
                        if scale == 'index':
                            higher_better = 'maximize' in col_meta and col_meta['maximize']
                            recalc_results = pd.DataFrame(index=db.index)
                            # needs to be split into calculations for each DS TASK ENV COMBO (due to different reference values)
                            for _, sub_db in db.groupby(['dataset', 'task', 'environment']):
                                ref_idx = sub_db[col].idxmax() # reference value always has highest index (== 1)
                                ref_val = value_db.loc[ref_idx,col]
                                recalc_results.loc[sub_db.index,f'{col}_train_pred'] = all_results[-1].loc[sub_db.index,f'{col}_train_pred'].map(lambda v: index_to_value(v, ref_val, higher_better))
                                recalc_results.loc[sub_db.index,f'{col}_test_pred'] = all_results[-1].loc[sub_db.index,f'{col}_test_pred'].map(lambda v: index_to_value(v, ref_val, higher_better))
                                recalc_results.loc[sub_db.index,f'{col}_train_err'] = value_db.loc[sub_db.index,col] - recalc_results.loc[sub_db.index,f'{col}_train_pred']
                                recalc_results.loc[sub_db.index,f'{col}_test_err'] = value_db.loc[sub_db.index,col] - recalc_results.loc[sub_db.index,f'{col}_test_pred']
                            all_results.append(recalc_results)
                            result_cols.append(f'{use_env}__rec_index')
                        print(f'{ft_name:<8} - {col:<18} - {str(db[cols].shape):<10} - Best Model: {sorted_models.index[0]:<17} - {error_info_as_string(all_results[-1], col)}')
                    if scale == 'index':
                        # recalculate the compound index score
                        index_pred = pd.DataFrame(index=db.index)
                        for split in ['_train_pred', '_test_pred']:
                            results = pd.concat([all_results[res_idx][f'{col}{split}'] for col, res_idx in compound_col_res_idc.items()], axis=1)
                            results = results.rename(lambda c: c.replace(split, ''), axis=1)
                            index_pred[f'compound_index{split}'] = [calculate_single_compound_rating(vals, custom_weights=weights) for _, vals in results.iterrows()]
                            index_pred[f'compound_index{split.replace("_pred", "_err")}'] = db['compound_index'] - index_pred[f'compound_index{split}']
                        print(f'{ft_name:<8} - {col:<18} - {str(db[cols].shape):<10} -                               - {error_info_as_string(index_pred, "compound_index")}')
                        all_results.append(index_pred)
                        result_cols.append(f'{use_env}__index')


            final_results = pd.concat(all_results, keys=result_cols, axis=1)
            final_results.to_pickle(os.path.join('exp_results', 'meta_learning', f'{ft_name}.pkl'))
