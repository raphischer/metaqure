import argparse
import os

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

from strep.load_experiment_logs import assemble_database
from strep.util import format_software, format_hardware, write_json
from methods import BUDGET_FILE

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

DB_DIR = 'exp_results/databases'
DB_COMPLETE = os.path.join(DB_DIR, f'metaqure.pkl')
DB_BL = os.path.join(DB_DIR, f'baselines.pkl')
DB_SUB = os.path.join(DB_DIR, f'subset.pkl')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Classification training with Tensorflow, based on PyTorch training")
    # data and model input
    parser.add_argument("--output-dir", default='/data/d1/sus-meta-results/logs/ws27_240605', type=str, help="path to saved outputs")
    parser.add_argument("--merged-dir", default='exp_results/logs')

    args = parser.parse_args()

    basename = os.path.basename(args.output_dir)
    mergedir = os.path.join(args.merged_dir, basename)
    db_file = os.path.join(DB_DIR, f'{basename}.pkl')

    database = assemble_database(args.output_dir, mergedir, None, PROPERTIES)
    if not os.path.isdir(DB_DIR):
        os.makedirs(DB_DIR)
    fixed = pd.read_pickle('exp_results/databases/svm_knn.pkl')
    for (model, ds), data in fixed.groupby(['model', 'dataset']):
        database.loc[(database['model'] == model) & (database['dataset'] == ds),'parameters'] = data['parameters'].values
        database.loc[(database['model'] == model) & (database['dataset'] == ds),'fsize'] = data['fsize'].values
    database.to_pickle(db_file)

    # load all dbs
    dbs = []
    for fname in sorted(os.listdir(DB_DIR)):
        if '.pkl' in fname and fname not in [os.path.basename(fname) for fname in [DB_COMPLETE, DB_BL, DB_SUB]]:
            dbs.append( pd.read_pickle(os.path.join(DB_DIR, fname)) )
            # print(f'{fname:<20} no nan {str(dbs[-1].dropna().shape):<12} original {str(dbs[-1].shape):<12}')

    # merge all dbs and do some small fixes
    complete = pd.concat(dbs)
    complete['parameters'] = complete['parameters'].fillna(0) # nan parameters are okay (occurs for PFN)
    complete = complete.dropna().reset_index(drop=True) # get rid of failed PFN evals
    # for some weird AGL experiments, codecarbon logged extreeemely high and unreasonable consumed energy (in the thousands and even millions of Watt)
    # we discard these outliers (assuming a max draw of 400 Watt) and do a simple kNN gap filling
    complete.loc[complete['train_power_draw'] / complete['train_running_time'] > 400,'train_power_draw'] = np.nan
    complete.loc[complete['power_draw'] / complete['running_time'] > 400,'power_draw'] = np.nan
    imputer = KNNImputer(n_neighbors=10, weights="uniform")
    numeric = complete.select_dtypes('number').columns
    complete.loc[:,numeric] = imputer.fit_transform(complete[numeric])
    
    # split into metaqure, baselines and subset
    baselines = complete[complete['model'].isin(['PFN', 'AGL', 'NAM', 'PFN4', 'PFN16', 'PFN64', 'PFN32'])]
    complete = complete.drop(baselines.index, axis=0)
    baselines.reset_index(drop=True).to_pickle(DB_BL)
    complete.reset_index(drop=True).to_pickle(DB_COMPLETE)
    subset = complete[complete['dataset'].isin(pd.unique(complete['dataset'])[5:15].tolist() + ['credit-g'])]
    subset.reset_index(drop=True).to_pickle(DB_SUB)

    # calculate budgets
    budgets = {}
    for (env, ds), data in complete.groupby(['environment', 'dataset']):
        env = env.split(' - ')[0]
        if env not in budgets:
            budgets[env] = {}
        budgets[env][ds] = data['train_running_time'].sum()
    write_json(BUDGET_FILE, budgets)

    # print some statistics
    for env, data in complete.groupby('environment'):
        print(env)
        print('  complete:', data.dropna().shape)
        print('  baselines:', baselines[baselines['environment'] == env].dropna().shape)
        print(f'  time per baseline: {sum(budgets[env.split(" - ")[0]].values())}')
