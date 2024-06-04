import argparse

import time
import os
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import ParameterSampler, RandomizedSearchCV, cross_val_score

from strep.util import write_json
from data_loading import data_variant_loaders
from methods import CLSF, get_budget


def hyperparam_fname(ds_name, method, outdir='./exp_results/hyperparameters'):
    return os.path.join(outdir, f'hyperparameters__{ds_name}__{method.replace(" ", "_")}.json')


def init_with_best_hyperparams(ds_name, method, seed, n_jobs, output_dir):
    if method == 'PFN': # use PFN baseline model
        from tabpfn import TabPFNClassifier
        return (None, TabPFNClassifier(device='cpu', seed=seed), None, lambda m: np.nan), np.nan

    clf = CLSF[method]
    # fix baseline method parameters that do not have the standard API
    if method == 'AGL':
        clf[1].set_params(**{'time_limit': get_budget(output_dir, ds_name)})
    if method == 'NAM':
        clf[1].timeout = max(get_budget(output_dir, ds_name), 20)
    if method == 'ASK':
        clf[1].set_params(**{'time_left_for_this_task': max(get_budget(output_dir, ds_name), 30), 'seed': seed})#, 'n_jobs': args.n_jobs})#, 'metric': 'accuracy'})

    # make sure to set parameters to the classifier, in case of pipeline with scaler
    clf_to_param = clf[1].steps[1][1] if hasattr(clf[1], 'steps') else clf[1]
    try: # check for hyperparam file
        fname = hyperparam_fname(ds_name, method)
        with open(fname, 'r') as hyperf:
            hyper_content = json.load(hyperf)
        best_rank_idx = np.argmin(hyper_content['rank_test_score'])
        best_params = hyper_content['params'][best_rank_idx]
        clf_to_param.set_params(**best_params)
        sensitivity = np.std(hyper_content['mean_test_score'])
    except FileNotFoundError:
        print('  no hyperparameter search information found, using default hyperparameters instead')
        sensitivity = np.nan
    # if possible, set n jobs and random state
    try:
        clf_to_param.set_params(**{'n_jobs': n_jobs})
    except ValueError:
        print('n_jobs cannot be set for method', method)
    try:
        clf_to_param.set_params(**{'random_state': seed})
    except ValueError:
        print('random_state cannot be set for method', method)
    
    return clf, sensitivity
    

def custom_hyperparam_search(method, X, y, outfile, n_iter, time_budget, seed, multiprocess, cv=5):
    _, clf, cls_params, _ = CLSF[method]
    clf_to_param = clf.steps[1][1] if hasattr(clf, 'steps') else clf
    n_jobs = cv if multiprocess else None
    # the easy way, without time budget
    if time_budget < 0:
        clf = RandomizedSearchCV(clf, cls_params, random_state=seed, n_iter=n_iter, verbose=6, cv=cv, n_jobs=n_jobs)
        clf.fit(X, y) # run the search
        results = clf.cv_results_
    else: # use a custom search and stop after elapsed time budget
        param_list = list(ParameterSampler(cls_params, n_iter=n_iter, random_state=seed))
        results = {'params': param_list, 'mean_test_score': [], 'std_test_score': []}
        t0 = time.time()
        for params in tqdm(param_list):
            clf_to_param.set_params(**params)
            try:
                scores = cross_val_score(clf, X, y, cv=cv, n_jobs=n_jobs)
                results['mean_test_score'].append(np.mean(scores))
                results['std_test_score'].append(np.std(scores))
            except Exception:
                results['mean_test_score'].append(np.nan)
                results['std_test_score'].append(np.nan)
            if (time.time() - t0) / 60 > time_budget:
                print('  - killed due to time limit!')
                break
        # calculate ranks
        results['rank_test_score'] = pd.DataFrame(results['mean_test_score']).fillna(0).rank(ascending=False).astype(int).iloc[:,0].tolist()

    write_json(outfile, results)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iter", default=50)
    parser.add_argument("--time-budget", default=50) # minutes
    parser.add_argument("--data-home", default="/data/d1/sus-meta-results/data")
    parser.add_argument("--ds", default='lung_cancer')
    parser.add_argument("--method", default='RF')
    parser.add_argument("--subsample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Seed to use")
    parser.add_argument("--multiprocess", default=True)
    parser.add_argument("--override", default=False)
    args = parser.parse_args()

    variant_loaders = data_variant_loaders(args.ds, args.data_home, args.seed, args.subsample)
    for ds_variant in variant_loaders:
        X, _, y, _, _, ds_name = ds_variant()
        outfile = hyperparam_fname(ds_name, args.method)
        if args.override or not os.path.isfile(outfile):
            print(f'Searching {args.method} hyperparameters for {outfile}')
            custom_hyperparam_search(args.method, X, y, outfile, args.n_iter, args.time_budget, args.seed, args.multiprocess)    

        # t0 = time.time()
        # custom_hyperparam_search(args.method, X_train, y_train, outfile)
        # t1 = time.time()
        # custom_hyperparam_search(args.method, X_train, y_train, outfile, multiprocess=True)
        # t2 = time.time()
        # print(f'{ds[:20]} SINGLE CORE SEARCH {(t1-t0)/60:4.3f} MULTI CORE SEARCH {(t2-t1)/60:4.3f}')
