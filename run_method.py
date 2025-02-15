import argparse
from datetime import timedelta
import os
import pickle
import time

import numpy as np
from sklearn.metrics import accuracy_score, top_k_accuracy_score, f1_score, precision_score, recall_score

from strep.util import create_output_dir, write_json, read_json
from strep.monitoring import init_monitoring
from data_loading import data_variant_loaders, ds_name_to_subsample
from run_hyperparam_search import init_with_best_hyperparams


CLSF_METRICS = {
    'accuracy': accuracy_score,
    'f1': lambda y1, y2, lab: f1_score(y1, y2, labels=lab, average='weighted'),
    'precision': lambda y1, y2, lab: precision_score(y1, y2, labels=lab, average='weighted'),
    'recall': lambda y1, y2, lab: recall_score(y1, y2, labels=lab, average='weighted'),
}


def score(y_pred, y_proba, y_test, classes):
    try:
        metrics = {}
        # calculating predictive quality metrics
        for score, func in CLSF_METRICS.items():
            try: # some score metrics need information on available classes
                metrics[score] = func(y_test, y_pred, classes)
            except TypeError:
                metrics[score] = func(y_test, y_pred)
        try:
            if classes.size == 2:
                y_proba = y_proba[:, 1]
            metrics['top_5_accuracy'] = top_k_accuracy_score(y_test, y_proba, k=5, labels=classes)
        except Exception:
            metrics['top_5_accuracy'] = metrics['accuracy'] # top5 is bounded by top1    
        return metrics
    except Exception:
        return {}

# additional metrics: generalization? hyperparameter sensitivity? hyperparameter fitting effort? model size?


def finalize_model(clf, output_dir, param_func, sensitivity):
    try:
        model_fname = os.path.join(output_dir, 'model.pkl')
        try:
            hyperparams = clf.steps[1][1].get_params() if hasattr(clf, 'steps') else clf.get_params()
        except Exception:
            hyperparams = 0
        with open(model_fname, 'wb') as modelfile:
            pickle.dump(clf, modelfile)
        return {
            'hyperparams': hyperparams,
            'params': param_func(clf),
            'fsize': os.path.getsize(model_fname),
            'hyperparam_sensitivity': sensitivity,
        }
    
    except Exception:
        return {}


def evaluate_single(ds_loader, args):
    print(f'Running evaluation on {args.ds} (subsample {args.subsample}) for {args.method} (seed - {args.seed})')
    t0 = time.time()
    X_train, X_test, y_train, y_test, args.feature_names, args.ds = ds_loader()
    args.subsample, args.ds_orig = ds_name_to_subsample(args.ds)
    output_dir = create_output_dir(args.output_dir, 'train', args.__dict__)
    (_, clf, _, param_func), sensitivity = init_with_best_hyperparams(args.ds, args.method, args.seed, args.n_jobs, output_dir)
    if args.subsample is not None:
        args.ds = args.ds.split('___')[1]

    ############## TRAINING ##############
    energy_tracker = init_monitoring(args.monitor_interval, output_dir)
    try:
        clf.fit(X_train, y_train)
    except (ValueError, AssertionError) as e: # can happen with PFN models
        clf, results = None, {}
        print(e)
    energy_tracker.stop()

    ############## PREDICT ##############
    if clf is not None:
        energy_tracker = init_monitoring(args.monitor_interval, output_dir)
        try:
            y_proba = clf.predict_proba(X_test)
        except Exception:
            y_proba = None
            y_pred = clf.predict(X_test)
        energy_tracker.stop()

        if y_proba is not None: # we also need the non-proba predictions, but unfair to compute them during profiling
            y_pred = clf.predict(X_test)

        # write results
        classes = np.unique(y_train) if not hasattr(clf, 'classes_') else clf.classes_
        results = {
            'history': {}, # TODO track history
            'model': finalize_model(clf, output_dir, param_func, sensitivity),
            'metrics': score(y_pred, y_proba, y_test, classes),
            'data': {'shape': {'train': X_train.shape, 'test': X_test.shape}}
        }
    write_json(os.path.join(output_dir, f'results.json'), results)

    ############## FNALIZE ##############

    print(f"Evaluation finished in {timedelta(seconds=int(time.time() - t0))} seconds, results can be found in {output_dir}\n")
    time.sleep(1)
    return output_dir


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Classification training with Tensorflow, based on PyTorch training")
    # data and model input
    parser.add_argument("--data-home", default="/data/d1/sus-meta-results/data")
    parser.add_argument("--ds", default="lung_cancer")
    parser.add_argument("--subsample", default=None)
    parser.add_argument("--method", default="RF")
    parser.add_argument("--n-jobs", default=-1)
    # output
    parser.add_argument("--output-dir", default='logs/sklearn', type=str, help="path to save outputs")
    parser.add_argument("--monitor-interval", default=.1, type=float, help="Setting to > 0 activates profiling every X seconds")
    # randomization and hardware
    parser.add_argument("--seed", type=int, default=42, help="Seed to use")

    args = parser.parse_args()

    variant_loaders = data_variant_loaders(args.ds, args.data_home, args.seed, args.subsample)
    for ds_variant in variant_loaders:
        evaluate_single(ds_variant, args)

        # sizes_file = os.path.join(os.getcwd(), 'dataset_split_sizes.json')
        # stored_sizes = read_json(sizes_file) if os.path.isfile(sizes_file) else {}
        # X_train, X_test, y_train, y_test, _, ds = ds_variant()
        # if ds in stored_sizes:
        #     if X_train.shape != tuple(stored_sizes[ds]["train"]) or X_test.shape != tuple(stored_sizes[ds]["test"]):
        #         print(f"DS {ds:<50} - already found shapes", stored_sizes[ds]["train"], stored_sizes[ds]["test"], "but they do not match with the loaded data shapes!", X_train.shape, X_test.shape)
        # else:
        #     print(f"Storing sizes for {ds}")
        #     stored_sizes[ds] = {"train": X_train.shape, "test": X_test.shape}
        #     write_json(sizes_file, stored_sizes)
