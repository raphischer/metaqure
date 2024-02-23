import argparse
import time

import numpy as np

from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier, kernels
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from scipy.stats import uniform

from data_loading import DATASETS, load_data

CLSF = {
    "kNN": (
        'k-Nearest Neighbors',
        KNeighborsClassifier(algorithm='auto'),
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
        SVC(), 
        {
            'kernel': ('linear', 'rbf', 'poly', 'sigmoid'),
            'C': uniform(0, 2),
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
            "max_depth": [10, 5],
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
            "max_depth": [10, 5],
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
            'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
            "algorithm": ['SAMME', 'SAMME.R']
        },
        # n_params = 2 * number of nodes (feature & threshold)
        lambda clf: sum([tree.tree_.node_count * 2 for tree in clf.estimators_])
    ),

    "GNB": (
        'Gaussian Naive Bayes',
        GaussianNB(),
        {
            "var_smoothing": [1e-6, 1e-9, 1e-12]
        },
        lambda clf: sum([clf.class_prior_.size, clf.epsilon_, ])
    ),

    "RR": (
        'Ridge Regression',
        linear_model.RidgeClassifier(),
        {
            'alpha': uniform(0, 2)
        },
        lambda clf: sum([clf.coef_.size, clf.intercept_.size])
    ),

    "LR": (
        'Logistic Regression',
        linear_model.LogisticRegression(max_iter=500),
        {
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'C': uniform(0, 2),
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
            'alpha': uniform(0, 2)
        },
        lambda clf: sum([clf.coef_.size, clf.intercept_.size])
    ),

    "MLP": (
        'Multilayer Perceptron',
        MLPClassifier(max_iter=500, early_stopping=True),
        {
            "hidden_layer_sizes" : [ (200,), (100,), (50,), (100, 50,), (80, 50,) ],
            "solver": ['sgd', 'adam'],
            "alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1],
            "learning_rate_init": [0.00001, 0.0001, 0.001, 0.01, 0.1],
        },
        lambda clf: sum([layer_w.size for layer_w in clf.coefs_] + [layer_i.size for layer_i in clf.intercepts_])
    )

    # "Gauss Process": (
    #     GaussianProcessClassifier(),
    #     {
    #         "kernel": [
    #             kernels.Matern(length_scale=1.0, nu=0.5),
    #             kernels.Matern(length_scale=1.0, nu=1.5),
    #             kernels.Matern(length_scale=1.0, nu=2.5),
    #             kernels.Matern(length_scale=0.5, nu=0.5),
    #             kernels.Matern(length_scale=0.5, nu=1.5),
    #             kernels.Matern(length_scale=0.5, nu=2.5),
    #             kernels.Matern(length_scale=2.0, nu=0.5),
    #             kernels.Matern(length_scale=2.0, nu=1.5),
    #             kernels.Matern(length_scale=2.0, nu=2.5),
    #             kernels.RBF(length_scale=1.0),
    #             kernels.RBF(length_scale=0.5),
    #             kernels.RBF(length_scale=2.0),
    #             kernels.ConstantKernel(constant_value=1.0),
    #             kernels.ConstantKernel(constant_value=0.5),
    #             kernels.ConstantKernel(constant_value=2.0),
    #             kernels.RationalQuadratic(length_scale=1.0, alpha=1.0),
    #             kernels.RationalQuadratic(length_scale=1.0, alpha=0.5),
    #             kernels.RationalQuadratic(length_scale=1.0, alpha=2.0),
    #             kernels.RationalQuadratic(length_scale=0.5, alpha=1.0),
    #             kernels.RationalQuadratic(length_scale=0.5, alpha=0.5),
    #             kernels.RationalQuadratic(length_scale=0.5, alpha=2.0),
    #             kernels.RationalQuadratic(length_scale=2.0, alpha=1.0),
    #             kernels.RationalQuadratic(length_scale=2.0, alpha=0.5),
    #             kernels.RationalQuadratic(length_scale=2.0, alpha=2.0),
    #             kernels.ExpSineSquared(length_scale=1.0, periodicity=1.0),
    #             kernels.ExpSineSquared(length_scale=1.0, periodicity=0.5),
    #             kernels.ExpSineSquared(length_scale=1.0, periodicity=2.0),
    #             kernels.ExpSineSquared(length_scale=0.5, periodicity=1.0),
    #             kernels.ExpSineSquared(length_scale=0.5, periodicity=0.5),
    #             kernels.ExpSineSquared(length_scale=0.5, periodicity=2.0),
    #             kernels.ExpSineSquared(length_scale=2.0, periodicity=1.0),
    #             kernels.ExpSineSquared(length_scale=2.0, periodicity=0.5),
    #             kernels.ExpSineSquared(length_scale=2.0, periodicity=2.0),
    #         ],
    #         'n_restarts_optimizer': [0, 1, 2, 3, 4, 5]
    #     },
    #     lambda clf: 0
    # )
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-home", default="/data/d1/sus-meta-results/data")
    parser.add_argument("--ds", default='all')
    parser.add_argument("--method", default='RR')
    args = parser.parse_args()

    all_ds = DATASETS if args.ds.lower() == 'all' else [args.ds]
    short = args.method
    name, clf, cls_params, _ = CLSF[short]

    for ds in all_ds:
        X_train, X_test, y_train, y_test, feat = load_data(ds, args.data_home)
        t0 = time.time()
        clf.fit(X_train, y_train)
        t1 = time.time()
        score = clf.score(X_test, y_test)
        t2 = time.time()
        tr_s, te_s, n_class = y_train.size,  y_test.size, np.unique(y_test).size
        print(f'{ds[:10]:<10} {tr_s + te_s:>6} ({tr_s / (tr_s + te_s) * 100:4.1f}% train) instances  {n_class:>4} classes  {len(feat):>7} feat - {short:<4} accuracy {score*100:4.1f}%, training took {t1-t0:6.2f}s and scoring {t2-t1:6.2f}s')
    
    print('All methods:')
    print(' '.join(f'"{m}"' for m in CLSF.keys()))