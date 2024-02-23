import argparse
import os

import pandas as pd
import numpy as np
from seedpy import fixedseed

from sklearn.dummy import DummyRegressor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, max_error
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor

from data_loading import DATASETS


SCORING = {
    'MAE': mean_absolute_error,
    'MaxE': max_error
}
CV_SCORER = f'test_{next(iter(SCORING.keys()))}'


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
    'FriedmanTree':             (DecisionTreeRegressor, {'max_depth': 5, 'criterion': 'friedman_mse'}),
    'PoissonTree':              (DecisionTreeRegressor, {'max_depth': 5, 'criterion': 'poisson'})
}


def print_cv_scoring_results(model_name, scoring, scores):
    results = {} # 'fit time': (f'{np.mean(scores["fit_time"]):7.2f}'[:6], f'{np.std(scores["fit_time"]):6.1f}'[:5])}
    for split in ['train', 'test']:
        for score in scoring:
            res = scores[split + '_' + score]
            mean_res, std_res = np.mean(res), np.std(res)
            results[f'{split:<5} {score}'] = (f'{mean_res:7.5f}'[:6], f'{std_res:6.4f}'[:5])
    print(f'{model_name:<20}' + ' - '.join([f'{metric:<10} {mean_v} +- {std_v}' for metric, (mean_v, std_v) in results.items()]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classification training with Tensorflow, based on PyTorch training")
    parser.add_argument("--meta-feature-dir", default='meta_features')
    parser.add_argument("--seed", type=int, default=42, help="Seed to use")
    args = parser.parse_args()

    with fixedseed(np, seed=args.seed):

        col = 'accuracy'

        for meta_ft_file in os.listdir(args.meta_feature_dir):
            if not '.csv' in meta_ft_file:
                continue
            
            # load meta features
            meta_features = pd.read_csv(os.path.join(args.meta_feature_dir, meta_ft_file)).dropna().set_index('Unnamed: 0')
            meta_ft_cols = list(meta_features.columns)

            # store meta feature in database
            database = pd.read_pickle('exp_results/databases/ws28_240222.pkl')
            for ds, sub_db in database.groupby('dataset'):
                if ds in meta_features.index:
                    database.loc[sub_db.index,meta_ft_cols] = meta_features[meta_features.index == ds].values
            database['model_name'] = database['model']
            database['model'] = LabelEncoder().fit_transform(database['model'].values)
            meta_ft_cols.append('model')

            # prepare grouped cross-validation
            database = database.dropna()
            X, y = database[meta_ft_cols], database[col].values
            ft_encoding = {'model': 'cat'}
            for ft in X.columns:
                ft_encoding[ft] = 'num'
            group_info = LabelEncoder().fit_transform(database['dataset'].values) # split CV across datasets
            cv_splitted = list(GroupKFold().split(np.zeros((database.shape[0], 1)), None, group_info))
                
            predictions, true, proba = {}, {}, {}
            best_models, best_name, best_score, best_scores = None, '', np.inf, None
            split_index = np.zeros((X.shape[0], 1))

            print(f'\n\n\n\n:::::::::::::::: META LEARN {col} USING {meta_ft_file.split(".")[0]} - SHAPE {X.shape} \n')
            for model_name, (model_cls, params) in REGRESSORS.items():
                if model_cls not in [DummyRegressor, LinearRegression, SVR]:
                    params['random_state'] = args.seed
                clsf = model_cls(**params)
                # for models with intercept, onehot enocded features need to have one column dropped due to collinearity
                # https://stackoverflow.com/questions/44712521/very-large-values-predicted-for-linear-regression
                categories = [ sorted(pd.unique(X[feat]).tolist()) for feat, enc in ft_encoding.items() if enc == 'cat' ]
                config = ('first', categories) if hasattr(clsf, 'fit_intercept') else (None, categories)
                # create the feature preprocessing pipeline, with different encoders per enc type
                transformers = {}
                for enc_type in ft_encoding.values():
                    if enc_type not in transformers:
                        transformers[enc_type] = (enc_type, ENCODERS[enc_type](config), [ft for ft, enc_ in ft_encoding.items() if enc_ == enc_type])
                preprocessor = ColumnTransformer(transformers=list(transformers.values()))

                predictions[model_name] = np.zeros_like(y, dtype=float)
                true[model_name] = np.zeros_like(y, dtype=float)
                proba[model_name] = np.zeros_like(y, dtype=float)
                
                # init scoring dict
                scores = {}
                for score in SCORING.keys():
                    scores[f'train_{score}'] = []
                    scores[f'test_{score}'] = []
                
                # fit and predict for each split
                models = []
                for split_idx, (train_idx, test_idx) in enumerate(cv_splitted):
                    split_index[test_idx] = split_idx
                    X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]
                    model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', clsf)])
                    model.fit(X_train, y_train)
                    y_train_pred = model.predict(X_train)
                    # safe the predictions per model for later usage
                    predictions[model_name][test_idx] = model.predict(X_test)
                    true[model_name][test_idx] = y_test
                    if hasattr(model, "predict_proba"):
                        proba[model_name] = model.predict_proba(X_test)
                    # calculate scoring
                    for score_name, score in SCORING.items():
                        scores[f'train_{score_name}'].append(score(y_train, y_train_pred))
                        scores[f'test_{score_name}'].append(score(y_test, predictions[model_name][test_idx]))
                    models.append(model)

                # print scoring and best method
                print_cv_scoring_results(model_name, SCORING.keys(), scores)
                if np.mean(scores[CV_SCORER]) < np.mean(best_score):
                    best_models = models
                    best_name = model_name
                    best_score = scores[CV_SCORER]
                    best_scores = scores
            print('----------- BEST METHOD:')
            print_cv_scoring_results(best_name, SCORING.keys(), best_scores)

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