import argparse
import os

import pandas as pd

from strep.load_experiment_logs import assemble_database
from strep.util import format_software, format_hardware


PROPERTIES = {
    'meta': {
        'task': lambda log: log['directory_name'].split('_')[0],
        'dataset': lambda log: log['config']['ds'],
        'model': lambda log: log['config']['method'],
        'architecture': lambda log: format_hardware(log['execution_platform']['Processor']),
        'software': lambda log: format_software('Scikit-learn', log['requirements'])
    },

    'train': {
        'train_running_time': lambda log: log['emissions']['duration']['0'],
        'train_power_draw': lambda log: log['emissions']['energy_consumed']['0'] * 3.6e6,
        'running_time': lambda log: log['emissions']['duration']['1'] / log['results']['data']['shape']['test'][0],
        'power_draw': lambda log: log['emissions']['energy_consumed']['1'] * 3.6e6 / log['results']['data']['shape']['test'][0],
        'parameters': lambda log: log['results']['model']['params'],
        'fsize': lambda log: log['results']['model']['fsize'],
        'accuracy': lambda log: log['results']['metrics']['accuracy'],
        'f1': lambda log: log['results']['metrics']['f1'],
        'precision': lambda log: log['results']['metrics']['precision'],
        'recall': lambda log: log['results']['metrics']['recall'],
    },
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Classification training with Tensorflow, based on PyTorch training")
    # data and model input
    parser.add_argument("--output-dir", default='logs/sklearn', type=str, help="path to saved outputs")
    parser.add_argument("--db-dir", default='exp_results/databases')
    parser.add_argument("--merged-dir", default='exp_results/logs')

    args = parser.parse_args()

    basename = os.path.basename(args.output_dir)
    mergedir = os.path.join(args.merged_dir, basename)
    db_file = os.path.join(args.db_dir, f'{basename}.pkl')

    database = assemble_database(args.output_dir, mergedir, None, PROPERTIES)
    if not os.path.isdir(args.db_dir):
        os.makedirs(args.db_dir)
    database.to_pickle(db_file)

    dbs = [pd.read_pickle(os.path.join(args.db_dir, fname)) for fname in os.listdir(args.db_dir) if '.pkl' in fname and fname not in ['complete.pkl', 'baselines.pkl', 'subset.pkl']]
    complete = pd.concat(dbs).reset_index()
    baselines = complete[complete['model'].isin(['PFN4', 'PFN16', 'PFN64'])]
    complete = complete.drop(baselines.index, axis=0)

    baselines.reset_index().to_pickle(os.path.join(args.db_dir, f'baselines.pkl'))
    complete.reset_index().to_pickle(os.path.join(args.db_dir, f'complete.pkl'))

    subset = complete[complete['dataset'].isin(pd.unique(complete['dataset'])[5:15])]
    subset.reset_index().to_pickle(os.path.join(args.db_dir, f'subset.pkl'))
