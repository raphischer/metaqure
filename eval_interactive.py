import argparse
import os

from strep.index_and_rate import rate_database, find_relevant_metrics, load_database
from strep.util import load_meta
from strep.elex.app import Visualization


def preprocess_database(fname):
    # load database
    database = load_database(fname)
    for ds, data in database.groupby('dataset'):
        if data.dropna().shape != (20, 20):
            print(f'{ds:<60} {str(data.dropna().shape)}')
    # load meta infotmation
    meta = load_meta()
    # rate database
    database, metrics, xaxis_default, yaxis_default = find_relevant_metrics(database, meta)
    rated_database, boundaries, real_boundaries, references = rate_database(database, meta)
    return rated_database, meta, metrics, xaxis_default, yaxis_default, boundaries, real_boundaries, references


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default='exp_results/databases/complete.pkl')
    parser.add_argument("--host", default='localhost', type=str, help="default host") # '0.0.0.0'
    parser.add_argument("--port", default=8888, type=int, help="default port")
    parser.add_argument("--debug", default=False, type=bool, help="debugging")
    args = parser.parse_args()

    db = { 'DB': preprocess_database(args.database) }
    app = Visualization(db)
    app.run_server(debug=args.debug, host=args.host, port=args.port)