import argparse
import os
import sys

import pandas as pd

from strep.index_and_rate import rate_database, find_relevant_metrics, load_database
from strep.util import load_meta

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default='stats', choices=['stats', 'interactive', 'paper_results'])
    parser.add_argument("--db-dir", default='exp_results/databases')
    # interactive exploration params
    parser.add_argument("--host", default='localhost', type=str, help="default host") # '0.0.0.0'
    parser.add_argument("--port", default=8888, type=int, help="default port")
    parser.add_argument("--debug", default=False, type=bool, help="debugging")

    args = parser.parse_args()

    from data_loading import load_data
    import numpy as np
    db = load_database('exp_results/databases/ws28_240222.pkl')
    res = []
    for ds, data in db.groupby('dataset'):
        _, _, y, _, _ = load_data(ds, '/data/d1/sus-meta-results/data')
        n_classes = np.unique(y).size
        ref_acc = 1 / n_classes
        amax, amin = data["accuracy"].max(), data["accuracy"].min()
        diff = amax - ref_acc
        res.append( (diff, amax, amin, ref_acc, ds, n_classes) )
    for diff, amax, amin, ref_acc, ds, n_classes in sorted(res):
        print(f'{ds[:40]:<40} n classes {n_classes:<4} min acc {amin:6.4f} max acc {amax:6.4f} ref acc {ref_acc:6.4f} acc diff {diff:6.4f}')

    # load databases
    databases = []
    for fname in os.listdir(args.db_dir):
        fname = os.path.join(args.db_dir, fname)
        print('LOADING', fname)
        if not os.path.isfile(fname):
            raise RuntimeError('Could not find', fname)
        databases.append( load_database(fname) )
    database = pd.concat(databases)

    if args.mode == 'stats':
        for config, data in database.groupby('dataset'):
            print(f'{config:<50} {data.shape}, {data.dropna().shape}')
        sys.exit(0)

    meta = load_meta(os.path.dirname(fname))
    database, metrics, xaxis_default, yaxis_default = find_relevant_metrics(database, meta)
    rated_database, boundaries, real_boundaries, references = rate_database(database, meta, indexmode='best')

    print(f'    database {name} has {rated_database.shape} entries')
    databases[name] = ( rated_database, meta, metrics, xaxis_default, yaxis_default, boundaries, real_boundaries, references )

    if args.mode == 'interactive':
       from strep.elex.app import Visualization
       app = Visualization(databases, index_mode=INDEXMODE)
       app.run_server(debug=args.debug, host=args.host, port=args.port)

    if args.mode == 'paper_results':
        from paper_results import create_all
        create_all(databases)