import pandas as pd
import numpy as np

dbs = {
    'ws3 single': pd.read_pickle('exp_results/databases/ws3_240221.pkl'),
    'ws3 multi': pd.read_pickle('exp_results/databases/ws3_240220.pkl'),
    'ws28 single': pd.read_pickle('exp_results/databases/ws28_240120.pkl'),
    'ws28 multi': pd.read_pickle('exp_results/databases/ws28_240219.pkl'),
}

for db in dbs.values():
    db['parameters'] = db['parameters'] / 1e6
    db['fsize'] = db['fsize'] / 1e6
    db['train_power_draw'] = db['train_power_draw'] / 1e3

rel_cols = list(next(iter(dbs.values())).select_dtypes('number').columns)

for col in rel_cols:
    res = '    '.join([f'{key}: {db[col].dropna().mean():6.2f} (+- {db[col].dropna().std():6.1f})' for key, db in dbs.items()])
    print(f'{col:<20} {res}')


# diffs = []
# for keys, data1 in singlecore.groupby(['dataset', 'model']):
#     ds, mod = keys
#     data2 = multi_core[(multi_core['dataset'] == ds) & (multi_core['model'] == mod)]
#     res = {'model': mod, 'dataset': ds}
#     for col in rel_cols:
#         diff = data1[col].dropna().values - data2[col].dropna().values
#         res[col] = np.nan if diff.size < 1 else diff[0]
#     diffs.append( res )

# diffs = pd.DataFrame(diffs)
# for col in rel_cols:
#     for mod in pd.unique(diffs['model']):
#         diffs_ = diffs[diffs['model'] == mod][col]
#         print(f"{col:<25} {mod:<4} {np.mean(np.abs(diffs_)):5.3f} +- {np.std(np.abs(diffs_)):5.3f}")
# print(1)