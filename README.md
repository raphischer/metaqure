# MetaQuRe: Meta-Learning from Model Quality and Resource Consumption

Code and results for our associated research paper, which is [published at ECML PKDD 2024 (Research Track)](https://link.springer.com/chapter/10.1007/978-3-031-70368-3_13). Our work and data allows to meta-learn from model quality and resource consumption, which benefits sustainability and resource-awareness in automated machine learning!

## Interactive Exploration
You can dive into our assembled **MetaQuRe** data set via our [interactive exploration tool](https://strep.onrender.com/?database=MetaQuRe) - no code needs to be run locally!
Note that this webpage is [work in progress](https://github.com/raphischer/strep) and subject to change, so you might encounter delays, off-times, and slight differences to our paper.

## Accessing **MetaQuRe** data
If you want to explore our data from code, you can easily load it via
```python
from strep.index_and_rate import load_database, rate_database
from run_log_processing import DB_COMPLETE
database = load_database(DB_COMPLETE)

# run the following to also inspect the index-scaled results (will take some time, faster version in progress)
from strep.util import load_meta, prop_dict_to_val
meta_info = load_meta()
rated_db, _, _, _ = rate_database(database, meta_info)
index_only_db = prop_dict_to_val(rated_db, 'index')
```

## Repository Structure
- The complete **MetaQuRe** data can be loaded as a single [pandas dataframe](./exp_results/databases/metaqure.pkl)
- The [strep](./strep) library contains code for performing index-scaling, as well as our exploration tool. It was created as part of our work on [Sustainable and Trustworthy Reporting](https://github.com/raphischer/strep).
- Our [experimental results](./exp_results) contain the individual result databases of **MetaQuRe**, the hyperparameters for all algorithms, the meta-features for our data sets, and the meta-learning predictions for all feature sets (stored as individual dataframes)
- All experiments and evaluations can be performed with the top level `run_` scripts (`.py` performs single experiments, `.sh` runs multiple experiments)
- There are additional scripts for loading data and methods, as well as a `json` file with meta information on the measures (properties)

## Installation
Create a suitable Python environment via

```
conda create --name metaqure python==3.10
conda activate metaqure
pip install -r requirements.txt
```

## Jetson
If you want to perform experiments on AGX Jetson (like we did), you cannot use `codecarbon` for profiling the resource consumption. So instead of installing `codecarbon`, make sure to properly [setup up jetson-stats](https://github.com/rbonghi/jetson_stats). Our code should autodetect which profiling is available based on the installed libraries.

## Usage
You can replicate our experiments or assemble MetaQuRe for your own custom environment by running the individual scripts. Pass data and log directories depending on your own directory management. We performed experiments in the following order (note that this will likely take several days!):
```
bash run_method_algos.sh $datadir $logdir
bash run_method_baselines.sh $datadir $logdir
python run_log_processing.py --output-dir $logdir
python run_meta_learning.py
```
After that, you can either explore the results [interactively](./eval_interactive.py), or [replicate our paper results](./eval_paper_results.py).

**! BEING SUSTAINABLE IS A COMMUNITY EFFORT !**

If you are interested in replicating our experiments and making these results available, feel free to get in touch - we will gladly include them and highlight you as contributor!

## Citation
If you appreciate our work and code, please cite [our paper](https://link.springer.com/chapter/10.1007/978-3-031-70368-3_13) as given by Springer:

Fischer, R., Wever, M., Buschjäger, S., Liebig, T. (2024). MetaQuRe: Meta-learning from Model Quality and Resource Consumption. In: Bifet, A., Davis, J., Krilavičius, T., Kull, M., Ntoutsi, E., Žliobaitė, I. (eds) Machine Learning and Knowledge Discovery in Databases. Research Track. ECML PKDD 2024. Lecture Notes in Computer Science(), vol 14947. Springer, Cham. https://doi.org/10.1007/978-3-031-70368-3_13

or using the bibkey below:
```
@InProceedings{10.1007/978-3-031-70368-3_13,
author="Fischer, Raphael
and Wever, Marcel
and Buschj{\"a}ger, Sebastian
and Liebig, Thomas",
editor="Bifet, Albert
and Davis, Jesse
and Krilavi{\v{c}}ius, Tomas
and Kull, Meelis
and Ntoutsi, Eirini
and {\v{Z}}liobait{\.{e}}, Indr{\.{e}}",
title="{MetaQuRe}: Meta-learning from Model Quality and Resource Consumption",
booktitle="Machine Learning and Knowledge Discovery in Databases. Research Track",
year="2024",
publisher="Springer Nature Switzerland",
address="Cham",
pages="209--226",
isbn="978-3-031-70368-3"
}
```

© Raphael Fischer

