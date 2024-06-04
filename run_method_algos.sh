#!/bin/bash

# conda deactivate
# conda activate ramimp

for s in "42" "7803" "5321"
do
    for m in "RF" "XRF" "AB" "GNB" "RR" "SGD" "MLP" "kNN" "LR" "SVM"
    do
        # real-world data
        for d in "lung_cancer" "zoo" "iris" "hepatitis" "wine" "bone_marrow_transplant_children" "parkinsons" "breast_cancer_wisconsin_prognostic" "connectionist_bench_sonar_mines_vs_rocks" "image_segmentation" "glass_identification" "statlog_heart" "breast_cancer" "heart_failure_clinical_records" "heart_disease" "ecoli" "ionosphere" "dermatology" "olivetti_faces" "cirrhosis_patient_survival_prediction" "congressional_voting_records" "wholesale_customers" "arrhythmia" "thoracic_surgery_data" "kc2" "climate-model-simulation-crashes" "monks-problems-1" "wdbc" "ilpd" "monks-problems-2" "hcv_data" "balance_scale" "student_performance" "credit_approval" "breast_cancer_wisconsin_original" "national_poll_on_healthy_aging_npha" "blood-transfusion-service-center" "diabetes" "regensburg_pediatric_appendicitis" "glioma_grading_clinical_and_mutation_features" "statlog_vehicle_silhouettes" "raisin" "tic-tac-toe" "credit-g" "qsar-biodeg" "pc1" "diabetic_retinopathy_debrecen" "hill-valley" "banknote-authentication" "contraceptive_method_choice" "yeast" "amazon-commerce-reviews" "pc3" "one-hundred-plants-texture" "one-hundred-plants-shape" "car_evaluation" "steel-plates-fault" "kc1" "estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition" "aids_clinical_trials_group_study_175" "Titanic" "national_health_and_nutrition_health_survey_2013-2014_nhanes_age_prediction_subset" "ozone-level-8hr" "kr-vs-kp" "Bioresponse" "rice_cammeo_and_osmancik" "abalone" "predict_students_dropout_and_academic_success" "spambase" "Satellite" "phoneme" "optical_recognition_of_handwritten_digits" "statlog_landsat_satellite" "wine_quality" "isolet" "mushroom" "SpeedDating" "support2" "ai4i_2020_predictive_maintenance_dataset" "electrical_grid_stability_simulated_data" "pen-based_recognition_of_handwritten_digits" "online_shoppers_purchasing_intention_dataset" "dry_bean_dataset" "eeg-eye-state" "magic_gamma_telescope" "letter_recognition" "default_of_credit_card_clients" "bank-marketing" "adult" "statlog_shuttle" "mnist_784" "sepsis_survival_minimal_clinical_records"
        do
            timeout 3600s python run_method.py --ds $d --method $m --seed $s --data-home $1 --output-dir $2
        done

        # synthetic data
        for d in "make_classification_500_10_8_1_4_2_1-0" "make_classification_1000_20_10_10_5_2_1-0" "make_classification_2000_30_5_5_6_3_1-0" "make_classification_500_40_10_5_5_10_1-0" "make_classification_1000_50_10_10_5_2_0-7" "make_classification_2000_60_5_5_6_3_0-7" "make_classification_5000_50_40_5_30_1_1-3" "make_classification_10000_70_50_10_50_1_1-3" "make_classification_20000_50_30_5_15_2_1-3" "make_classification_60000_30_10_5_20_3_0-9" "make_circles_200_0-3_0-7" "make_circles_800_0-2_0-8" "make_moons_500_0-3" "make_moons_900_0-5" "make_hastie_10_2_1000"
        do
            timeout 3600s python run_method.py --ds $d --method $m --seed $s --data-home $1 --output-dir $2
        done

        # real-world subset data
        for d in "lung_cancer" "connectionist_bench_sonar_mines_vs_rocks" "student_performance" "credit_approval" "qsar-biodeg" "spambase" "bank-marketing"
        do
            timeout 3600s python run_method.py --ds $d --method $m --seed $s --data-home $1 --output-dir $2 --subsample 2
        done

        for d in "credit-g" "hill-valley" "one-hundred-plants-texture" "one-hundred-plants-shape" "ozone-level-8hr" "kr-vs-kp" "optical_recognition_of_handwritten_digits" "support2"
        do
            timeout 3600s python run_method.py --ds $d --method $m --seed $s --data-home $1 --output-dir $2 --subsample 3
        done

        for d in "olivetti_faces" "cirrhosis_patient_survival_prediction" "arrhythmia" "regensburg_pediatric_appendicitis" "amazon-commerce-reviews" "Bioresponse" "isolet" "mushroom" "SpeedDating" "adult" "mnist_784"
        do
            timeout 3600s python run_method.py --ds $d --method $m --seed $s --data-home $1 --output-dir $2 --subsample 5
        done
    done
done

python run_log_processing.py --output-dir $2