def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import argparse
import os
import time

import pandas as pd
import numpy as np
from seedpy import fixedseed

from sklearn.dummy import DummyRegressor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, max_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor

from strep.index_and_rate import rate_database, load_database, index_to_value, calculate_single_compound_rating
from strep.util import load_meta, prop_dict_to_val

from run_meta_feature_extraction import load_meta_features
from data_loading import ds_cv_split
from run_log_processing import DB_COMPLETE

SCORING = {
    'MAE': mean_absolute_error,
    'MaxE': max_error
}
CV_SCORER = 'test_MAE'

ENCODERS = {
    'num': lambda config: Pipeline(steps=[ ('scaler', StandardScaler()) ]),
    'cat': lambda config: Pipeline(steps=[ ('onehot', OneHotEncoder(drop=config[0], categories=config[1])) ])
}

ML_RES_DIR = os.path.join('exp_results', 'meta_learning')

REGRESSORS = {
    # 'Global Mean':              (DummyRegressor, {}),
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


def evaluate_regressor(regr, X, y, cv_splitted, seed, scale):
    results = pd.DataFrame(index=X.index, columns=['train_pred', 'test_pred', 'train_err', 'test_err'])
    model_cls, params = REGRESSORS[regr]
    if model_cls not in [DummyRegressor, LinearRegression, SVR]:
        params['random_state'] = seed
    clsf = model_cls(**params)
    drop = 'first' if hasattr(clsf, 'fit_intercept') else None
    preprocessor = ColumnTransformer(transformers=[
        ('cat', Pipeline(steps=[ ('onehot', OneHotEncoder(drop=drop)) ]), ['environment']),
        ('num', Pipeline(steps=[ ('scaler', StandardScaler()) ]), X.drop('environment', axis=1).columns.tolist())
    ])
    # fit and predict for each split
    for split_idx, (train_idx, test_idx) in enumerate(cv_splitted):
        X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
        model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', clsf)])
        model.fit(X_train, y_train)
        results.loc[X.iloc[train_idx].index,f'train_pred{split_idx}'] = model.predict(X_train)
        results.loc[X.iloc[test_idx].index,'test_pred'] = model.predict(X_test)
    results['train_pred'] = results[[c_ for c_ in results.columns if 'train_pred' in c_]].mean(axis=1)
    results = results.drop([f'train_pred{split}' for split in range(len(cv_splitted))], axis=1)
    if scale == 'index': 
        results['train_pred'] = results['train_pred'].clip(lower=0, upper=1)
        results['test_pred'] = results['test_pred'].clip(lower=0, upper=1)
    results['train_err'], results['test_err'] = results['train_pred'] - y, results['test_pred'] - y
    return results


def recalculate_original_values(results, ref_vals, value_db, col, col_meta):
    higher_better = 'maximize' in col_meta and col_meta['maximize']
    recalc_results = pd.DataFrame(index=results.index)
    recalc_results['train_pred'] = index_to_value(results['train_pred'], ref_vals, higher_better)
    recalc_results['test_pred'] = index_to_value(results['test_pred'], ref_vals, higher_better)
    recalc_results['train_err'] = value_db.loc[results.index,col] - recalc_results['train_pred']
    recalc_results['test_err'] = value_db.loc[results.index,col] - recalc_results['test_pred']
    return recalc_results


def error_info_as_string(row, col):
    return ' - '.join([f'{c.replace(f"{col}_", "")}: {row[c].abs().mean():10.6f} +- {row[c].abs().std():10.2f}' for c in row.columns if 'err' in c])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Seed to use")
    args = parser.parse_args()

    all_meta_features = load_meta_features()
    meta = load_meta()
    weights = {col: val['weight'] for col, val in meta['properties'].items()}
    database = load_database(DB_COMPLETE)
    rated_db = rate_database(database, meta, indexmode='best')[0]
    best_regr = {}
    
    with fixedseed(np, seed=args.seed):
        for ft_name, meta_features in all_meta_features.items():
            # load meta features and database
            meta_ft_cols = list(meta_features.columns) + ['environment']
            all_results = {'index': pd.DataFrame(index=database.index), 'value': pd.DataFrame(index=database.index), 'recalc_value': pd.DataFrame(index=database.index)}
            index_db, value_db = prop_dict_to_val(rated_db, 'index'), prop_dict_to_val(rated_db, 'value')
            # already collect reference values (only done once, for faster recalculation)
            for (d, t, e), sub_db in index_db.groupby(['dataset', 'task', 'environment']):
                for col in meta['properties'].keys():
                    ref_idx = sub_db[col].idxmax() # reference value always has highest index (== 1)
                    index_db.loc[(index_db['dataset'] == d) & (index_db['task'] == t) & (index_db['environment'] == e),f'{col}_ref_val'] = value_db.loc[ref_idx,col]
            for db, scale in zip([index_db, value_db], ['index', 'value']):
                # store meta feature in respective dataframe
                for ds, sub_db in db.groupby('dataset'):
                    if ds in meta_features.index:
                        db.loc[sub_db.index,meta_features.columns] = meta_features[meta_features.index == ds].values
                # train meta-learners for each individual algorithm
                for algo, algo_db in db.groupby('model'):
                    # prepare grouped cross-validation
                    cv_splits = ds_cv_split(algo_db['dataset'], n_splits=5)
                    opt_find_db = algo_db.iloc[cv_splits[0][0]] # first cv train split used for finding optimal model choice
                    opt_find_cv_splits = ds_cv_split(opt_find_db['dataset'], n_splits=5)
                    print(f'\n\n\n\n:::::::::::::::: META LEARN FOR {algo} USING {ft_name} with {scale}\n')
                    for col, col_meta in meta['properties'].items():
                        # 1. find optimal model choice on subset of the data (first cv train split)
                        regr_results = {}
                        for regr in REGRESSORS.keys():
                            t0 = time.time()
                            res = evaluate_regressor(regr, opt_find_db[meta_ft_cols], opt_find_db[col], opt_find_cv_splits, args.seed, scale)
                            t1 = time.time()
                            if scale == 'index': # make the selection based on MAE of REAL measurements
                                res = recalculate_original_values(res, opt_find_db[f'{col}_ref_val'], value_db, col, col_meta)
                                # print(f'  evaluated    {regr:<20} {t1-t0:5.3f}  recalculated {regr:<20} {time.time()-t1:5.3f}')
                            # else:
                            #     print(f'  evaluated    {regr:<20} {t1-t0:5.3f}')
                            regr_results[regr] = res
                        sorted_results = list(sorted([(res['test_err'].abs().mean(), regr) for regr, res in regr_results.items()]))
                        best_model = sorted_results[0][1]
                        if best_model not in best_regr:
                            best_regr[best_model] = 0
                        best_regr[best_model] += 1
                        # TODO also store sorted_models.index[0] (best model name?)

                        # 2. train and evaluate best model on full data!
                        best_model_results = evaluate_regressor(best_model, algo_db[meta_ft_cols], algo_db[col], cv_splits, args.seed, scale)
                        best_results_renamed = best_model_results.rename(lambda c_ : f'{col}_{c_}', axis=1)
                        all_results[scale].loc[algo_db.index,best_results_renamed.columns] = best_results_renamed
                        if scale == 'index': # recalculate index predictions to real value predictions
                            recalc = recalculate_original_values(best_model_results, algo_db[f'{col}_ref_val'], value_db, col, col_meta).rename(lambda c_ : f'{col}_{c_}', axis=1)
                            all_results['recalc_value'].loc[algo_db.index,recalc.columns] = recalc
                        err_data = recalc if scale == 'index' else best_results_renamed
                        print(f'{ft_name:<8} - {col:<18} - {str(algo_db[meta_ft_cols].shape):<10} - Best Model: {best_model:<17} - {error_info_as_string(err_data, col)}')
                # recalculate the compound index score
                index_pred = pd.DataFrame(index=db.index)
                for split in ['_train_pred', '_test_pred']:
                    # retrieve the index predictions for all property columns
                    results = all_results['index'][[col for col in all_results['index'].columns if split in col]].rename(lambda c: c.replace(split, ''), axis=1)
                    index_pred[f'compound_index{split}'] = [calculate_single_compound_rating(vals, custom_weights=weights) for _, vals in results.iterrows()]
                    index_pred[f'compound_index{split.replace("_pred", "_err")}'] = db['compound_index'] - index_pred[f'compound_index{split}']
                print(f'{"ft_name":<8} - {"compound":<18} - {str(db[meta_ft_cols].shape):<10} -                               - {error_info_as_string(index_pred, "compound_index")}')
                all_results['index'].loc[:,index_pred.columns] = index_pred

            final_results = pd.concat(all_results.values(), keys=all_results.keys(), axis=1)
            final_results.to_pickle(os.path.join(ML_RES_DIR, f'{ft_name}.pkl'))

    print(best_regr)