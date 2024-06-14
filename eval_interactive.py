import argparse

from strep.index_and_rate import rate_database, find_relevant_metrics, load_database
from strep.util import load_meta
from strep.elex.app import Visualization

from run_log_processing import DB_COMPLETE

def preprocess_database(fname):
    # load database
    database = load_database(fname)
    # load meta infotmation
    meta = load_meta()
    # rate database
    database, metrics, xaxis_default, yaxis_default = find_relevant_metrics(database, meta)
    rated_database, boundaries, real_boundaries, references = rate_database(database, meta)
    return rated_database, meta, metrics, xaxis_default, yaxis_default, boundaries, real_boundaries, references


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default='localhost', type=str, help="default host") # '0.0.0.0'
    parser.add_argument("--port", default=8888, type=int, help="default port")
    parser.add_argument("--debug", default=False, type=bool, help="debugging")
    args = parser.parse_args()

    db = { 'DB': preprocess_database(DB_COMPLETE) }
    app = Visualization(db)
    app.run_server(debug=args.debug, host=args.host, port=args.port)