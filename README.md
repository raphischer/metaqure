# MetaQuRe: Meta-Learning from Model Quality and Resource Consumption

Code and results for the associated research paper (anonymized version for reviewers, paper currently under review).

## Interactive Exploration
Install the Python [libraries required](./requirements.txt) for visualization and run `python eval_interactive.py`. 
After the start up (which will take several minutes), the tool can be accessed from a [local webbrowser page](http://localhost:8888/).
If you want to explore our data from code, you can easily load it via
```python
from strep.index_and_rate import load_database, rate_database
from run_log_processing import DB_COMPLETE
database = load_database(DB_COMPLETE)

# run the following to also inspect the index-scaled results (will take some time)
from strep.util import load_meta, prop_dict_to_val
meta_info = load_meta()
rated_db, _, _, _ = rate_database(database, meta_info)
index_only_db = prop_dict_to_val(rated_db, 'index')
```

## Structure
- The complete MetaQuRe data can be loaded as a single [pandas dataframe](./exp_results/databases/complete.pkl)
- The [strep](./strep) library contains code for performing index-scaling, as well as our exploration tool
- Our [experimental results](./exp_results) contain the individual result databases of MetaQuRe, the hyperparameters for all algorithms, the meta-features for our data sets, and the meta-learning predictions for all feature sets (stored as individual dataframes)
- All experiments and evaluations can be performed with the top level `run_` scripts (`.py` performs single experiments, `.sh` runs multiple experiments)
- There are additional scripts for loading data and methods, as well as a `json` file with meta information on the measures (properties)

## Installation
Create a suitable Python environment created via

```
conda create --name metaqure python==3.10
conda activate metaqure
pip install -r requirements.txt
```

## Jetson
Instead of installing codecarbon, make sure to properly setup up [jetson-stats](https://github.com/rbonghi/jetson_stats).

## Usage
You can replicate our experiments or assemble MetaQuRe for your own environment by running the individual scripts. Pass data and log directories depending on your own directory management. We performed experiments in the following order (this will likely take several days!):
```
bash run_method_algos.sh $datadir $logdir
bash run_method_baselines.sh $datadir $logdir
python run_log_processing.py --output-dir $logdir
python run_meta_learning.py
```
After that, you can either explore the results [interactively](./eval_interactive.py), or [replicate our paper results](./eval_paper_results.py).