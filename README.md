
## Installation
The main environment can be created by running

```
conda create --name ramimp python==3.10
conda activate ramimp
pip install -r requirements.txt
```

Unfortunately, `auto-sklearn` is not compatible with `autogluon` as they require different versions of `scikit-learn`. For this reason, we utilize a second environment which can be created with

```
conda create --name ramimp_ask python==3.8
conda activate ramimp_ask
pip install -r requirements_ask.txt
```


## Usage

python data_loading.py --data-home /home/lfischer/data/susmeta/data

python run_method.py --data-home /home/lfischer/data/susmeta/data

https://askubuntu.com/questions/155791/how-do-i-sudo-a-command-in-a-script-without-being-asked-for-a-password