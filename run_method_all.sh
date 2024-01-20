#!/bin/bash

for d in "olivetti_faces" "arrhythmia" "kc2" "climate-model-simulation-crashes" "monks-problems-1" "wdbc" "ilpd" "monks-problems-2" "blood-transfusion-service-center" "diabetes" "tic-tac-toe" "credit-g" "qsar-biodeg" "pc1" "hill-valley" "banknote-authentication" "amazon-commerce-reviews" "pc3" "one-hundred-plants-texture" "one-hundred-plants-shape" "steel-plates-fault" "kc1" "Titanic" "ozone-level-8hr" "kr-vs-kp" "lfw_pairs" "Bioresponse" "spambase" "Satellite" "phoneme" "SpeedDating" "lfw_people" "eeg-eye-state" "20newsgroups_vectorized" "bank-marketing" "adult" "mnist_784" "covtype"
do
    for m in "kNN" "SVM" "RF" "XRF" "AB" "GNB" "RR" "LR" "SGD" "MLP"
    do
        timeout 3600s python run_method.py --ds $d --method $m --data-home $1 --output-dir $2
    done
done