#!/bin/bash

for m in "kNN" "SVM" "RF" "XRF" "AB" "GNB" "RR" "LR" "SGD" "MLP"
do
    for d in "lung_cancer" "zoo" "iris" "hepatitis" "wine" "bone_marrow_transplant:_children" "parkinsons" "breast_cancer_wisconsin_prognostic" "connectionist_bench_sonar,_mines_vs._rocks" "image_segmentation" "glass_identification" "statlog_heart" "breast_cancer" "heart_failure_clinical_records" "heart_disease" "soybean_large" "ecoli" "ionosphere" "dermatology" "olivetti_faces" "cirrhosis_patient_survival_prediction" "congressional_voting_records" "wholesale_customers" "arrhythmia" "thoracic_surgery_data" "kc2" "climate-model-simulation-crashes" "monks-problems-1" "wdbc" "ilpd" "monks-problems-2" "hcv_data" "balance_scale" "student_performance" "credit_approval" "breast_cancer_wisconsin_original" "national_poll_on_healthy_aging_npha" "blood-transfusion-service-center" "diabetes" "regensburg_pediatric_appendicitis" "glioma_grading_clinical_and_mutation_features" "statlog_vehicle_silhouettes" "raisin" "tic-tac-toe" "credit-g" "qsar-biodeg" "pc1" "diabetic_retinopathy_debrecen" "hill-valley" "banknote-authentication" "contraceptive_method_choice" "yeast" "amazon-commerce-reviews" "pc3" "one-hundred-plants-texture" "one-hundred-plants-shape" "myocardial_infarction_complications" "car_evaluation" "steel-plates-fault" "kc1" "estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition" "aids_clinical_trials_group_study_175" "Titanic" "national_health_and_nutrition_health_survey_2013-2014_nhanes_age_prediction_subset" "ozone-level-8hr" "kr-vs-kp" "Bioresponse" "rice_cammeo_and_osmancik" "abalone" "predict_students_dropout_and_academic_success" "spambase" "Satellite" "phoneme" "optical_recognition_of_handwritten_digits" "statlog_landsat_satellite" "wine_quality" "isolet" "mushroom" "SpeedDating" "support2" "ai4i_2020_predictive_maintenance_dataset" "electrical_grid_stability_simulated_data" "pen-based_recognition_of_handwritten_digits" "online_shoppers_purchasing_intention_dataset" "nursery" "dry_bean_dataset" "eeg-eye-state" "20newsgroups_vectorized" "magic_gamma_telescope" "letter_recognition" "default_of_credit_card_clients" "bank-marketing" "adult" "statlog_shuttle" "mnist_784" "diabetes_130-us_hospitals_for_years_1999-2008" "sepsis_survival_minimal_clinical_records" "cdc_diabetes_health_indicators" "covtype"
    do
        timeout 3600s python run_method.py --ds $d --method $m --data-home $1 --output-dir $2
    done

    for d in "lung_cancer" "connectionist_bench_sonar,_mines_vs._rocks" "student_performance" "credit_approval" "qsar-biodeg" "spambase" "bank-marketing" "covtype"
    do
        timeout 3600s python run_method.py --ds $d --method $m --data-home $1 --output-dir $2 --resample 2
    done

    for d in "credit-g" "hill-valley" "one-hundred-plants-texture" "one-hundred-plants-shape" "ozone-level-8hr" "kr-vs-kp" "optical_recognition_of_handwritten_digits" "support2"
    do
        timeout 3600s python run_method.py --ds $d --method $m --data-home $1 --output-dir $2 --resample 3
    done

    for d in "olivetti_faces" "cirrhosis_patient_survival_prediction" "arrhythmia" "regensburg_pediatric_appendicitis" "amazon-commerce-reviews" "myocardial_infarction_complications" "Bioresponse" "isolet" "mushroom" "SpeedDating" "20newsgroups_vectorized" "adult" "mnist_784" "diabetes_130-us_hospitals_for_years_1999-2008"
    do
        timeout 3600s python run_method.py --ds $d --method $m --data-home $1 --output-dir $2 --resample 3
    done
done

python process_logs.py --output-dir $2