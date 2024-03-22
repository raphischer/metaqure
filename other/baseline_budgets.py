import pandas as pd

from strep.util import write_json
from methods import BUDGET_FILE


if __name__ == "__main__":
    db = pd.read_pickle('exp_results/databases/complete.pkl')
    budgets = {}
    for (env, ds), data in db.groupby(['environment', 'dataset']):
        env = env.split(' - ')[0]
        if env not in budgets:
            budgets[env] = {}
        budgets[env][ds] = data['train_running_time'].sum()
    for env, vals in budgets.items():
        print(f'{env:<40} time per baseline: {sum(vals.values())}')
    write_json(BUDGET_FILE, budgets)
