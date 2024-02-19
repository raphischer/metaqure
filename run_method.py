import argparse
from datetime import timedelta
import json
import os
import pickle
import time
import sys
import re

import numpy as np
from sklearn.metrics import accuracy_score, top_k_accuracy_score, f1_score, precision_score, recall_score
from codecarbon import OfflineEmissionsTracker

from strep.util import fix_seed, create_output_dir, Logger, write_json

from methods import CLSF
from data_loading import load_data
from run_hyperparam_search import init_with_best_hyperparams


CLSF_METRICS = {
    'accuracy': accuracy_score,
    'f1': lambda y1, y2: f1_score(y1, y2, average='micro'),
    'precision': lambda y1, y2: precision_score(y1, y2, average='micro'),
    'recall': lambda y1, y2: recall_score(y1, y2, average='micro'),
}


def score(y_pred, y_proba, y_test, clf):
    metrics = {}
    # calculating predictive quality metrics
    for score, func in CLSF_METRICS.items():
        try: # some score metrics need information on available classes
            metrics[score] = func(y_test, y_pred, labels=clf.classes_)
        except TypeError:
            metrics[score] = func(y_test, y_pred)
    if y_proba is not None:
        
        if clf.classes_.size == 2:
            y_proba = y_proba[:, 1]
        metrics['top_5_accuracy'] = top_k_accuracy_score(y_test, y_proba, k=5, labels=clf.classes_)
    else:
        metrics['top_5_accuracy'] = metrics['accuracy'] # top5 is bounded by top1
    return metrics

# additional metrics: generalization? hyperparameter sensitivity? hyperparameter fitting effort? model size?


def finalize_model(clf, output_dir, param_func, sensitivity):
    model_fname = os.path.join(output_dir, 'model.pkl')
    with open(model_fname, 'wb') as modelfile:
        pickle.dump(clf, modelfile)    

    # count flops of infering single random data row
    # test_data = np.random.rand(1, clf.n_features_in_)
    # flops = monitor_flops_papi(lambda : clf.predict(test_data))[0]

    clf_info = {
        'hyperparams': clf.get_params(),
        'params': param_func(clf),
        'fsize': os.path.getsize(model_fname),
        'hyperparam_sensitivity': sensitivity
        # 'flops': flops
    }
    return clf_info


def evaluate_single(args):
    print(f'Running evaluation on {args.ds} for {args.method}')
    t0 = time.time()
    args.seed = fix_seed(args.seed)

    ############## TRAINING ##############
    output_dir = create_output_dir(args.output_dir, 'train', args.__dict__)
    tmp = sys.stdout # reroute the stdout to logfile, remember to call close!
    sys.stdout = Logger(os.path.join(output_dir, f'logfile.txt'))

    X_train, X_test, y_train, y_test, feature_names = load_data(args.ds)
    (_, clf, _, param_func), sensitivity = init_with_best_hyperparams(args.ds, args.method)

    # train
    emissions_tracker = OfflineEmissionsTracker(measure_power_secs=args.monitor_interval, log_level='warning', country_iso_code="DEU", save_to_file=True, output_dir=output_dir)
    emissions_tracker.start()
    clf.fit(X_train, y_train)
    emissions_tracker.stop()

    # predict
    emissions_tracker = OfflineEmissionsTracker(measure_power_secs=args.monitor_interval, log_level='warning', country_iso_code="DEU", save_to_file=True, output_dir=output_dir)
    emissions_tracker.start()
    if hasattr(clf, 'predict_proba'):
        y_proba = clf.predict_proba(X_test)
    else:
        y_proba = None
        y_pred = clf.predict(X_test)
    emissions_tracker.stop()
    if y_proba is not None:
        y_pred = clf.predict(X_test)

    # write results
    results = {
        'history': {}, # TODO track history
        'model': finalize_model(clf, output_dir, param_func, sensitivity),
        'metrics': score(y_pred, y_proba, y_test, clf),
        'data': {'shape': {'train': X_train.shape, 'test': X_test.shape}}
    }
    write_json(os.path.join(output_dir, f'results.json'), results)

    ############## FNALIZE ##############

    print(f"Evaluation finished in {timedelta(seconds=int(time.time() - t0))} seconds, results can be found in {output_dir}\n")
    sys.stdout.close()
    sys.stdout = tmp
    return output_dir


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Classification training with Tensorflow, based on PyTorch training")
    # data and model input
    parser.add_argument("--data-home", default=None)
    parser.add_argument("--ds", default="credit-g")
    parser.add_argument("--method", default="RR")
    # output
    parser.add_argument("--output-dir", default='logs/sklearn', type=str, help="path to save outputs")
    parser.add_argument("--monitor-interval", default=.01, type=float, help="Setting to > 0 activates profiling every X seconds")
    # randomization and hardware
    parser.add_argument("--seed", type=int, default=42, help="Seed to use (if -1, uses and logs random seed)")

    args = parser.parse_args()

    evaluate_single(args)
