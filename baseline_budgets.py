import os

import pandas as pd

from strep.util import read_json, write_json, format_hardware

BUDGET_FILE = 'exp_results/budget.json'

def get_budget(output_dir, ds, min_time=5):
    architecture = format_hardware(read_json(os.path.join(output_dir, 'execution_platform.json'))['Processor'])
    budgets = read_json(BUDGET_FILE)
    return max(int(budgets[architecture][ds]), min_time)


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
