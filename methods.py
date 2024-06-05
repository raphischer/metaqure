import argparse
import time
import os

import numpy as np

from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier, kernels
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from strep.util import read_json, write_json, format_hardware

from data_loading import load_data

BUDGET_FILE = 'exp_results/budget.json'


def get_budget(output_dir, ds, min_time=15):
    try:
        architecture = format_hardware(read_json(os.path.join(output_dir, 'execution_platform.json'))['Processor'])
        budgets = read_json(BUDGET_FILE)
        return max(int(budgets[architecture][ds]), min_time)
    except Exception:
        print(f'  no budget found for {architecture} {ds} - using min time {min_time}')
        return min_time
    

CLSF = {
    "kNN": (
        'k-Nearest Neighbors',
        make_pipeline(StandardScaler(), KNeighborsClassifier()),
        {
            'n_neighbors': [1, 3, 5, 10, 15, 20, 30, 50],
            'algorithm': ['auto', 'ball_tree', 'kd_tree'],
            'leaf_size': [10, 20, 30],
            'p': [1, 2, 3]

        },
        lambda clf: clf.n_features_in_ * clf.n_samples_fit_ 
    ),
    
    "SVM": (
        'Support Vector Machine',
        make_pipeline(StandardScaler(), SVC(cache_size=1000)),
        {
            'kernel': ('linear', 'rbf', 'poly', 'sigmoid'),
            'C': np.exp(np.random.rand((50)) * 6 - 3),
            'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1]
        },
        lambda clf: sum([clf.class_weight_.size, clf.intercept_.size, clf.support_vectors_.size])
    ),

    "RF": (
        'Random Forest',
        RandomForestClassifier(), 
        {
            "n_estimators": [10, 20, 40, 75, 100, 150],
            "criterion": ["gini", "entropy"],
            "max_depth": [10, 5, 3],
            "max_features": ['sqrt', 'log2', 5, 10, 20],
        },
        # n_params = 2 * number of nodes (feature & threshold)
        lambda clf: sum([tree.tree_.node_count * 2 for tree in clf.estimators_])
    ),

    "XRF": (
        'Extra Random Forest',
        ExtraTreesClassifier(), 
        {
            "n_estimators": [10, 20, 40, 75, 100, 150],
            "criterion": ["gini", "entropy"],
            "max_depth": [10, 5, 3],
            "max_features": ['sqrt', 'log2', 5, 10, 20],
        },
        # n_params = 2 * number of nodes (feature & threshold)
        lambda clf: sum([tree.tree_.node_count * 2 for tree in clf.estimators_])
    ),

    "AB": (
        'AdaBoost',
        AdaBoostClassifier(),
        {
            "n_estimators": [10, 20, 40, 75, 100, 150, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0],
            "algorithm": ['SAMME', 'SAMME.R']
        },
        # n_params = 2 * number of nodes (feature & threshold)
        lambda clf: sum([tree.tree_.node_count * 2 for tree in clf.estimators_])
    ),

    "GNB": (
        'Gaussian Naive Bayes',
        GaussianNB(),
        {
            "var_smoothing": [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]
        },
        lambda clf: sum([clf.class_prior_.size, clf.epsilon_, ])
    ),

    "RR": (
        'Ridge Regression',
        linear_model.RidgeClassifier(),
        {
            'alpha': np.exp(np.random.rand((50)) * 6 - 3)
        },
        lambda clf: sum([clf.coef_.size, clf.intercept_.size])
    ),

    "LR": (
        'Logistic Regression',
        linear_model.LogisticRegression(max_iter=500),
        {
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'C': np.exp(np.random.rand((50)) * 6 - 3),
            'solver': ['lbfgs', 'sag', 'saga'],
        },
        lambda clf: sum([clf.coef_.size, clf.intercept_.size])
    ),

    "SGD": (
        'Linear Stochastic Gradient Descent',
        linear_model.SGDClassifier(max_iter=500),
        {
            "loss" : ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error'],
            "penalty": ['l2', 'l1', 'elasticnet'],
            'alpha': np.exp(np.random.rand((50)) * 6 - 3)
        },
        lambda clf: sum([clf.coef_.size, clf.intercept_.size])
    ),

    "MLP": (
        'Multilayer Perceptron',
        MLPClassifier(max_iter=500, early_stopping=True),
        {
            "hidden_layer_sizes" : [ (200,), (100,), (50,), (50, 30,), (100, 30,), (80, 50,), (60, 40, 20,), ],
            "solver": ['sgd', 'adam'],
            "alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1],
            "learning_rate_init": [0.00001, 0.0001, 0.001, 0.05, 0.01, 0.05, 0.1, 0.2],
        },
        lambda clf: sum([layer_w.size for layer_w in clf.coefs_] + [layer_i.size for layer_i in clf.intercepts_])
    )
}

####### AUTOGLUON
try:
    from autogluon.tabular.experimental import TabularClassifier
    CLSF['AGL'] = (
        'AutoGluon',
        TabularClassifier(presets='medium_quality', path='/tmp/', verbosity=0),
        None,
        lambda clf: 0
    )
except ImportError:
    print('AutoGluon not available, please install: pip install autogluon==1.0.0')

####### NaiveAutoML
try:
    from naiveautoml import NaiveAutoML
    CLSF['NAM'] = (
        'NaiveAutoML',
        NaiveAutoML(scoring="accuracy"),
        None,
        lambda clf: 0
    )
except ImportError:
    print('NaiveAutoML not available, please install: pip install naiveautoml')

####### AUTOSKLEARN
# try:
#     from autosklearn.experimental.askl2 import AutoSklearn2Classifier
#     import pandas as pd
#     if not hasattr(pd.DataFrame, 'iteritems'): # be compatiable with pandas >= 2.0
#         pd.DataFrame.iteritems = pd.DataFrame.items
#     CLSF['ASK'] = (
#         'AutoSklearn',
#         AutoSklearn2Classifier(),
#         None,
#         lambda clf: 0
#     )
# except ImportError:
#     print('AutoSklearn not available, please install: pip install auto-sklearn==0.15')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-home", default="/data/d1/sus-meta-results/data")
    parser.add_argument("--ds", default='adult')
    parser.add_argument("--method", default='RR')
    args = parser.parse_args()
    short = args.method
    name, clf, cls_params, _ = CLSF[short]

    X_train, X_test, y_train, y_test, feat = load_data(args.ds, args.data_home)
    t0 = time.time()
    clf.fit(X_train, y_train)
    t1 = time.time()
    score = clf.score(X_test, y_test)
    t2 = time.time()
    tr_s, te_s, n_class = y_train.size,  y_test.size, np.unique(y_test).size
    print(f'{args.ds[:10]:<10} {tr_s + te_s:>6} ({tr_s / (tr_s + te_s) * 100:4.1f}% train) instances  {n_class:>4} classes  {len(feat):>7} feat - {short:<4} accuracy {score*100:4.1f}%, training took {t1-t0:6.2f}s and scoring {t2-t1:6.2f}s')
    
    print('All methods:')
    print(' '.join(f'"{m}"' for m in CLSF.keys()))