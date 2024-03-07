import argparse
import os
import re

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import pandas as pd

from seedpy import fixedseed
from sklearn import datasets as sk_datasets
from sklearn.model_selection import train_test_split, KFold, GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


# can be updated by running ucimlrepo.list_available_datasets
# res = {}
# for row in a.split('\n'):
#     match = re.match(r'(.*)  (\d+)\s*', row)
#     res[match.group(1).strip().lower().replace(' ', '_').replace('(', '').replace(')', '').replace("'", '')] = int(match.group(2))
UCI_MAP = {'abalone': 1, 'adult': 2, 'auto_mpg': 9, 'automobile': 10, 'balance_scale': 12, 'breast_cancer': 14,
           'breast_cancer_wisconsin_original': 15, 'breast_cancer_wisconsin_prognostic': 16,
           'breast_cancer_wisconsin_diagnostic': 17, 'car_evaluation': 19, 'census_income': 20, 'credit_approval': 27,
           'computer_hardware': 29, 'contraceptive_method_choice': 30, 'covertype': 31, 'dermatology': 33, 'ecoli': 39,
           'glass_identification': 42, 'heart_disease': 45, 'hepatitis': 46, 'image_segmentation': 50, 'ionosphere': 52,
           'iris': 53, 'isolet': 54, 'letter_recognition': 59, 'liver_disorders': 60, 'lung_cancer': 62, 'mushroom': 73,
           'nursery': 76, 'optical_recognition_of_handwritten_digits': 80, 'pen-based_recognition_of_handwritten_digits': 81,
           'solar_flare': 89, 'soybean_large': 90, 'spambase': 94, 'tic-tac-toe_endgame': 101,
           'congressional_voting_records': 105, 'wine': 109, 'yeast': 110, 'zoo': 111,
           'statlog_australian_credit_approval': 143, 'statlog_german_credit_data': 144, 'statlog_heart': 145,
           'statlog_landsat_satellite': 146, 'statlog_shuttle': 148, 'statlog_vehicle_silhouettes': 149,
           'connectionist_bench_sonar_mines_vs_rocks': 151, 'magic_gamma_telescope': 159, 'forest_fires': 162,
           'concrete_compressive_strength': 165, 'parkinsons': 174, 'wine_quality': 186, 'parkinsons_telemonitoring': 189,
           'bank_marketing': 222, 'ilpd_indian_liver_patient_dataset': 225,
           'individual_household_electric_power_consumption': 235, 'energy_efficiency': 242, 'banknote_authentication': 267,
           'bike_sharing_dataset': 275, 'thoracic_surgery_data': 277, 'wholesale_customers': 292,
           'diabetes_130-us_hospitals_for_years_1999-2008': 296, 'student_performance': 320,
           'diabetic_retinopathy_debrecen': 329, 'online_news_popularity': 332, 'default_of_credit_card_clients': 350,
           'online_retail': 352, 'air_quality': 360, 'online_shoppers_purchasing_intention_dataset': 468,
           'electrical_grid_stability_simulated_data': 471, 'real_estate_valuation': 477,
           'heart_failure_clinical_records': 519,
           'estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition': 544, 'rice_cammeo_and_osmancik': 545,
           'apartment_for_rent_classified': 555, 'seoul_bike_sharing_demand': 560, 'bone_marrow_transplant_children': 565,
           'hcv_data': 571, 'myocardial_infarction_complications': 579, 'ai4i_2020_predictive_maintenance_dataset': 601,
           'dry_bean_dataset': 602, 'predict_students_dropout_and_academic_success': 697,
           'glioma_grading_clinical_and_mutation_features': 759, 'sepsis_survival_minimal_clinical_records': 827,
           'raisin': 850, 'cirrhosis_patient_survival_prediction': 878, 'support2': 880,
           'national_health_and_nutrition_health_survey_2013-2014_nhanes_age_prediction_subset': 887,
           'aids_clinical_trials_group_study_175': 890, 'cdc_diabetes_health_indicators': 891,
           'infrared_thermography_temperature': 925, 'national_poll_on_healthy_aging_npha': 936,
           'regensburg_pediatric_appendicitis': 938           
}

# Sklearn real-life datasets
SKLEARN_DATASETS = [
    'olivetti_faces',
    # 'lfw_people', # TODO too hard / image data 
    # 'lfw_pairs', # TODO too hard / image data
    # '20newsgroups_vectorized',
    # 'covtype',
    # 'kddcup99', # TODO fix error
    # 'california_housing', # TODO fix error
    # too small / not stored locally:
    # 'breast_cancer',
    # 'digits',
    # 'iris',
    # 'wine'
]

# Popular OpenML datasets
OPENML_DATASETS = [
    # verified multiclass sorted by number of runs
    'credit-g',
    'blood-transfusion-service-center',
    'monks-problems-2',
    'tic-tac-toe',
    'monks-problems-1',
    'steel-plates-fault',
    'kr-vs-kp',
    'qsar-biodeg',
    'wdbc',
    'phoneme',
    'diabetes',
    'ozone-level-8hr',
    'hill-valley',
    'kc2',
    'eeg-eye-state',
    'climate-model-simulation-crashes',
    'spambase',
    'kc1',
    'ilpd',
    'pc1',
    'pc3',

    # additional verified multiclass sorted by number of likes (at least three)
    'SpeedDating',
    'mnist_784',
    'banknote-authentication',
    # 'adult', # UCI census_income
    'Titanic',
    'Satellite',
    'bank-marketing',

    # additional verified multiclass sorted by number of downloads (at least 40)
    'one-hundred-plants-texture',
    'arrhythmia',
    'amazon-commerce-reviews',
    'one-hundred-plants-shape',
    'Bioresponse',
]

UCI_DATASETS = [
    'abalone',
    'adult',
    'balance_scale',
    'breast_cancer',
    'breast_cancer_wisconsin_original',
    'breast_cancer_wisconsin_prognostic',
    # 'breast_cancer_wisconsin_diagnostic', # OPENML wdbc
    'car_evaluation',
    # 'census_income', # OPENML adult
    'credit_approval',
    'contraceptive_method_choice',
    'dermatology',
    'ecoli',
    'glass_identification',
    'heart_disease',
    'hepatitis',
    'image_segmentation',
    'ionosphere',
    'iris',
    'isolet',
    'letter_recognition',
    'lung_cancer',
    'mushroom',
    # 'nursery',
    'optical_recognition_of_handwritten_digits',
    'pen-based_recognition_of_handwritten_digits',
    # 'soybean_large',
    # 'tic-tac-toe_endgame', # OPENML tic-tac-toe
    'congressional_voting_records',
    'wine',
    'yeast',
    'zoo',
    # 'statlog_australian_credit_approval', # UCI credit_approval
    # 'statlog_german_credit_data', # OPENML credit-g
    'statlog_heart',
    'statlog_landsat_satellite',
    'statlog_shuttle',
    'statlog_vehicle_silhouettes',
    'connectionist_bench_sonar_mines_vs_rocks',
    'magic_gamma_telescope',
    'parkinsons',
    'wine_quality',
    # 'bank_marketing', # OPENML bank-marketing
    # 'ilpd_indian_liver_patient_dataset', # OPENML ilpd
    # 'energy_efficiency', # too hard to learn
    # 'banknote_authentication', # OPENML banknote-authentication
    'thoracic_surgery_data',
    'wholesale_customers',
    # 'diabetes_130-us_hospitals_for_years_1999-2008',
    'student_performance',
    'diabetic_retinopathy_debrecen',
    # 'online_news_popularity', # too hard to learn
    'default_of_credit_card_clients',
    # 'online_retail', # fetch crashes
    'online_shoppers_purchasing_intention_dataset',
    'electrical_grid_stability_simulated_data',
    'heart_failure_clinical_records',
    'estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition',
    'rice_cammeo_and_osmancik',
    # 'apartment_for_rent_classified', # fetch crashes
    'bone_marrow_transplant_children',
    'hcv_data',
    # 'myocardial_infarction_complications',
    'ai4i_2020_predictive_maintenance_dataset',
    'dry_bean_dataset',
    'predict_students_dropout_and_academic_success',
    'glioma_grading_clinical_and_mutation_features',
    'sepsis_survival_minimal_clinical_records',
    'raisin',
    'cirrhosis_patient_survival_prediction',
    'support2',
    'national_health_and_nutrition_health_survey_2013-2014_nhanes_age_prediction_subset',
    'aids_clinical_trials_group_study_175',
    # 'cdc_diabetes_health_indicators',
    'national_poll_on_healthy_aging_npha',
    'regensburg_pediatric_appendicitis'
]

def load_sklearn_feature_names(ds):
    if hasattr(ds, 'feature_names'):
        feat = ds.feature_names
    else:
        feat = np.array([f'feat_{idx}' for idx in range(ds.data.shape[1])])
    if not isinstance(feat, np.ndarray):
        return np.array(feat)
    return feat
    

def load_sklearn(ds_name, data_home=None):
    ds_loader = getattr(sk_datasets, f'fetch_{ds_name}') if hasattr(sk_datasets, f'fetch_{ds_name}') else getattr(sk_datasets, f'load_{ds_name}')
    ds = ds_loader(data_home=data_home)
    feature_names = load_sklearn_feature_names(ds)
    return ds.data, ds.target, feature_names


def load_openml(ds_name, data_home=None):
    data = sk_datasets.fetch_openml(name=ds_name, data_home=data_home, parser='auto')
    X = pd.get_dummies(data['data']).astype(float) # one-hot
    X, feature_names = X.values, X.columns.values
    y, cat = pd.factorize(data['target'])
    return X, y, feature_names


def load_uci(ds_name, data_home=None):
    fname = os.path.join(data_home, 'uci_local', f'{ds_name}.pkl')
    try:
        X = pd.read_pickle(fname)
    except FileNotFoundError:
        import ucimlrepo
        ds = ucimlrepo.fetch_ucirepo(id=UCI_MAP[ds_name])
        if 'Classification' not in ds['metadata']['tasks']:
            raise RuntimeError()
        X = pd.concat([ds.data.features, ds.data.targets], axis=1)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        X.to_pickle(fname)
    X, y = X.iloc[:,:-1], X.iloc[:, -1]
    X = pd.get_dummies(X).astype(float) # one-hot
    X, feature_names = X.values, X.columns.values
    y, cat = pd.factorize(y)
    return X, y, feature_names


def generate_sklearn(ds_name, seed):
    param_funcs = {
        'make_circles': lambda s: {'n_samples': int(s[0]), 'noise': float(s[1].replace('-', '.')), 'factor': float(s[1].replace('-', '.'))},
        'make_moons': lambda s: {'n_samples': int(s[0]), 'noise': float(s[1].replace('-', '.'))},
        'make_hastie_10_2': lambda s: {'n_samples': int(s[0])},
        'make_classification': lambda s: {'n_samples': int(s[0]), 'n_features': int(s[1]), 'n_informative': int(s[2]),
                                          'n_redundant': int(s[3]), 'n_classes': int(s[4]), 'n_clusters_per_class': int(s[5]),
                                          'class_sep': float(s[6].replace('-', '.')), 'shift': None, 'scale': None}
    }
    for name, param_func in param_funcs.items():
        if name in ds_name:
            func = getattr(sk_datasets, name)
            params = param_func(ds_name.replace(f'{name}_', '').split('_'))
            params['random_state'] = seed
            X, y = func(**params)
            feat = np.array([f'feat_{idx}' for idx in range(X.shape[1])])
            return X, y, feat


def subsample_to_ds_name(subsample, ds_name):
    return f'v{subsample[1]}_{subsample[0]}___{ds_name}'


def ds_name_to_subsample(ds_name):
    try:
        iter, n_var, ds_orig = re.match(r'v(\d)_(\d)___(.*)', ds_name).groups()
        return (int(iter), int(n_var)), ds_orig
    except AttributeError: # unsuccessful match
        return None, ds_name


def load_data(ds_name, data_home=None, seed=42, subsample=None):
    with fixedseed(np, seed=seed):
        if ds_name in SKLEARN_DATASETS:
            X, y, feature_names = load_sklearn(ds_name, data_home)
        elif ds_name in OPENML_DATASETS:
            X, y, feature_names = load_openml(ds_name, data_home)
        elif ds_name in UCI_DATASETS:
            X, y, feature_names = load_uci(ds_name, data_home)
        elif 'make_' in ds_name:
            X, y, feature_names = generate_sklearn(ds_name, seed)
        else:
            raise RuntimeError(f'Dataset {ds_name} not found!')
    
    X = SimpleImputer(missing_values=np.nan, strategy='median').fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

    if subsample is not None: # use a fixed seed to sample features into variants
        kf = [idc[1] for idc in KFold(n_splits=subsample[0], random_state=430286, shuffle=True).split(np.arange(len(feature_names)))]
        idc = kf[subsample[1]]
        X_train = X_train[:,idc]
        X_test = X_test[:,idc]
        feature_names = feature_names[idc]
        ds_name = subsample_to_ds_name(subsample, ds_name)

    return X_train, X_test, y_train, y_test, list(feature_names), ds_name


def data_variant_loaders(ds_name, data_home=None, seed=42, subsample=None):
    if subsample is None:
        return [lambda: load_data(ds_name, data_home, seed, subsample)]
    subsample = int(subsample)
    assert subsample > 1
    return [lambda n=n: load_data(ds_name, data_home, seed, (subsample, n)) for n in range(subsample)]


def ds_cv_split(input_ds=None, n_splits=5):
    if input_ds is None:
        input_ds = []
        for ds, subsample in input_ds:
            to_add = [ds] if subsample is None else [ subsample_to_ds_name((subsample, n), ds) for n in range(subsample)]
            input_ds = input_ds + to_add
    # split CV across original datasets
    ds_original = [ ds_name_to_subsample(ds)[1] for ds in input_ds ]
    group_info = LabelEncoder().fit_transform(ds_original)
    split_idc = list(GroupKFold(n_splits=n_splits).split(np.zeros((len(input_ds), 1)), None, group_info))
    return split_idc



GENERATED_DATASETS = [
    "make_classification_500_10_8_1_4_2_1-0", "make_classification_1000_20_10_10_5_2_1-0", "make_classification_2000_30_5_5_6_3_1-0",
    "make_classification_500_40_10_5_5_10_1-0", "make_classification_1000_50_10_10_5_2_0-7", "make_classification_2000_60_5_5_6_3_0-7",
    "make_classification_5000_50_40_5_30_1_1-3", "make_classification_10000_70_50_10_50_1_1-3", "make_classification_20000_50_30_5_15_2_1-3",
    "make_classification_80000_80_10_5_20_3_0-9",
    "make_circles_200_0-3_0-7", "make_circles_800_0-2_0-8", "make_moons_500_0-3", "make_moons_900_0-5", "make_hastie_10_2_1000"
]

DATASETS = UCI_DATASETS + SKLEARN_DATASETS + OPENML_DATASETS + GENERATED_DATASETS

SUBSAMPLE = {
    2: ["lung_cancer", "connectionist_bench_sonar_mines_vs_rocks", "student_performance", "credit_approval", "qsar-biodeg", "spambase", "bank-marketing"],
    3: ["credit-g", "hill-valley", "one-hundred-plants-texture", "one-hundred-plants-shape", "ozone-level-8hr", "kr-vs-kp", "optical_recognition_of_handwritten_digits", "support2"],
    5: ["olivetti_faces", "cirrhosis_patient_survival_prediction", "arrhythmia", "regensburg_pediatric_appendicitis", "amazon-commerce-reviews", "Bioresponse", "isolet", "mushroom", "SpeedDating", "adult", "mnist_784"]
}

ALL_DS = []
for ds in DATASETS:
    ALL_DS.append((ds, None))
for subsample, subds in SUBSAMPLE.items():
    for ds in subds:
        ALL_DS.append((ds, subsample))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-home", default="/data/d1/sus-meta-results/data")
    args = parser.parse_args()

    size_ds = []
    subsampleX2, subsampleX3, subsampleX5 = [], [], []


    # import plotly.express as px
    # from methods import CLSF
    # from sklearn.model_selection import cross_val_score
    # import time
    # data = {}
    # for ds in ['make_classification_500_10_8_1_4_2_1-0', 'make_classification_1000_20_10_10_5_2_1-0', 'make_classification_2000_30_5_5_6_3_1-0',
    #            'make_classification_500_40_10_5_5_10_1-0', 'make_classification_1000_50_10_10_5_2_0-7', 'make_classification_2000_60_5_5_6_3_0-7',
    #            'make_classification_5000_50_40_5_30_1_1-3', 'make_classification_10000_70_50_10_50_1_1-3', 'make_classification_20000_30_5_5_60_3_1-3',
    #            'make_classification_80000_80_10_5_20_3_0-9',
    #            'make_circles_200_0-3_0-7', 'make_circles_800_0-2_0-8', 'make_moons_500_0-3', 'make_moons_900_0-5', 'make_hastie_10_2_1000']:
    #     X_train, X_test, y_train, y_test, feat, _ = load_data(ds, args.data_home)
    #     data[ds] = X_train, y_train
    #     px.scatter(x=X_train[:, 0], y=X_train[:, 1], color=y_train, title=ds).show()
    # for ds, (X_train, y_train) in data.items():
    #     t1 = time.time()
    #     scores = {meth: np.mean(cross_val_score(CLSF[meth][1], X_train, y_train)) for meth in ['RR', 'RF', 'MLP']}
    #     print(f'{ds:<50} {str(X_train.shape):<12} {np.unique(y_train).size:<2} classes {time.time()-t1:7.3f}s - {" - ".join([f"{meth:<3} {scr:5.3f}%" for meth, scr in scores.items()])}')



    for ds in DATASETS:
        X_train, X_test, y_train, y_test, feat, _ = load_data(ds, args.data_home)
        tr_s, te_s, n_class = y_train.size,  y_test.size, np.unique(y_test).size
        if X_train.shape[1] > 100:
            subsampleX5.append(ds)
        elif X_train.shape[1] > 60:
            subsampleX3.append(ds)
        elif X_train.shape[1] > 40:
            subsampleX2.append(ds)
        print(f'{ds[:20]:<20} {tr_s + te_s:>6} ({tr_s / (tr_s + te_s) * 100:4.1f}% train) instances  {n_class:>4} classes  {len(feat):>7} feat - {str(feat)[:50]} ...')
        size_ds.append( (tr_s + te_s, ds) )

    print('Ordered by size:')
    print(' '.join([ f'"{ds}"' for _, ds in sorted(size_ds) ]))

    print('Subsamplable X2:')
    print(' '.join([ f'"{ds}"' for _, ds in sorted(size_ds) if ds in subsampleX2]))

    print('Subsamplable X3:')
    print(' '.join([ f'"{ds}"' for _, ds in sorted(size_ds) if ds in subsampleX3]))

    print('Subsamplable X5:')
    print(' '.join([ f'"{ds}"' for _, ds in sorted(size_ds) if ds in subsampleX5]))
