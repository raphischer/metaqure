import argparse

import time
import os
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import ParameterSampler, RandomizedSearchCV, cross_val_score

from data_loading import DATASETS, load_data
from methods import CLSF
from strep.util import write_json


def hyperparam_fname(ds_name, method, outdir='./hyperparameters'):
    return os.path.join(outdir, f'hyperparameters__{ds_name}__{method.replace(" ", "_")}.json')


def init_with_best_hyperparams(ds_name, method, n_jobs=-1, outdir='./hyperparameters'):
    clf = CLSF[method]
    fname = hyperparam_fname(ds_name, method, outdir)
    try:
        with open(fname, 'r') as hyperf:
            hyper_content = json.load(hyperf)
        best_rank = hyper_content['rank_test_score'].index(1)
        best_params = hyper_content['params'][best_rank]
        clf[1].set_params(**best_params)
        try:
            clf[1].set_params(**{'n_jobs': n_jobs})
        except ValueError:
            print('n_jobs cannot be set for method', method)
        sensitivity = np.std(hyper_content['mean_test_score'])
    except FileNotFoundError:
        print('  no hyperparameter search information found, using default hyperparameters instead')
        sensitivity = np.nan
    return clf, sensitivity
    

def custom_hyperparam_search(method, X, y, outfile, n_iter, time_budget, seed, multiprocess, cv=5):
    _, clf, cls_params, _ = CLSF[method]
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
            clf.set_params(**params)
            scores = cross_val_score(clf, X, y, cv=cv, n_jobs=n_jobs)
            results['mean_test_score'].append(np.mean(scores))
            results['std_test_score'].append(np.std(scores))
            if (time.time() - t0) / 60 > time_budget:
                print('  - killed due to time limit!')
                break
        # calculate ranks
        results['rank_test_score'] = pd.DataFrame(results['mean_test_score']).rank(method='min').astype(int).iloc[:,0].tolist()

    write_json(outfile, results)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iter", default=50)
    parser.add_argument("--time-budget", default=60) # minutes
    parser.add_argument("--data-home", default="/data/d1/sus-meta-results/data")
    parser.add_argument("--ds", default='adult')
    parser.add_argument("--method", default='AB')
    parser.add_argument("--subsample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Seed to use")
    parser.add_argument("--multiprocess", default=True)
    args = parser.parse_args()

    to_process = []

    if args.subsample is not None and args.subsample > 1:
        subsample = args.subsample
        for n in range(args.subsample):
            X_train, _, y_train, _, _ = load_data(args.ds, args.data_home, seed=args.seed, subsample=(subsample, n) )
            outfile = hyperparam_fname(f'v{n}___{args.ds}', args.method)
            to_process.append( (args.method, X_train, y_train, outfile, args.seed) )
    else:
        X_train, _, y_train, _, _ = load_data(args.ds, args.data_home, seed=args.seed)
        outfile = hyperparam_fname(args.ds, args.method)
        to_process.append( (args.method, X_train, y_train, outfile, args.seed) )
    
    for method, X_train, y_train, outfile, seed in to_process:
        print(f'Searching hyperparameters for {outfile}')
        custom_hyperparam_search(method, X_train, y_train, outfile, args.n_iter, args.time_budget, seed, args.multiprocess)

        # t0 = time.time()
        # custom_hyperparam_search(args.method, X_train, y_train, outfile)
        # t1 = time.time()
        # custom_hyperparam_search(args.method, X_train, y_train, outfile, multiprocess=True)
        # t2 = time.time()
        # print(f'{ds[:20]} SINGLE CORE SEARCH {(t1-t0)/60:4.3f} MULTI CORE SEARCH {(t2-t1)/60:4.3f}')


    # dir = create_output_dir(dir='sklearn_hyperparameters', prefix='hyperparameters')
    # n_jobs = 10
    
    # for ds_name in sel_datasets:
    #     X_train, X_test, y_train, y_test = load_data(ds_name)

    #     # #### TEST DATASET
    #     # clf = GaussianProcessClassifier()
    #     # clf.fit(X_train, y_train)
    #     # for split, X, y in [('train', X_train, y_train), ('test', X_test, y_test)]:
    #     #     pred = clf.predict(X)
    #     #     print(f'{ds_name:<25} {str(X_train.shape):<13} {split:<6} accuracy {accuracy_score(y, pred)*100:6.2f}')




    #     #### RANDOMSEARCH
    #     try:
    #         for name, (_, classifier, cls_params, _) in classifiers.items():
    #             print(f'Running hyperparameter search for {ds_name:<15} {name:<18}')
    #             # t_start = time.time()
    #             multithread_classifier = 'n_jobs' in classifier.get_params().keys()
    #             if multithread_classifier:
    #                 classifier.set_params(**{'n_jobs': n_jobs})
    #                 clf = RandomizedSearchCV(classifier, cls_params, random_state=0, n_iter=50, verbose=6, n_jobs=None)
    #             else:
    #                 clf = RandomizedSearchCV(classifier, cls_params, random_state=0, n_iter=50, verbose=6, n_jobs=n_jobs)
    #             search = clf.fit(X_train, y_train)
    #             with open(os.path.join(dir, f'hyperparameters__{ds_name}__{name.replace(" ", "_")}.json'), 'w') as outfile:
    #                 json.dump(clf.cv_results_, outfile, indent=4, cls=PatchedJSONEncoder)
    #     except Exception as e:
    #         print(e)



    #         # t_train_end = time.time()
    #         # result_scores = {'fit_time': t_train_end - t_start}
    #         # result_scores['inf_time'] = 0
    #         # for split, X, y in [('train', X_train, y_train), ('test', X_test, y_test)]:
    #         #     y_pred = classifier.predict(X)
    #         #     accuracy = accuracy_score(y, y_pred)
    #         #     result_scores[f'{split}_acc'] = accuracy * 100
    #         # result_scores['inf_time'] = time.time() - t_train_end
    #         # print(f'{ds_name:<15} {str(X_train.shape):<13} {str(X_test.shape):<13} {name:<18} ' + ' - '.join([f'{key} {val:6.2f}' for key, val in result_scores.items()]))
    #     # print('                :::::::::::::             ')
