#!/bin/bash

MODEL_NAME="cct_7_3x1_32"
MODEL_NAME_SHORT="cct"

START_FOLDER_NAME="avs_${MODEL_NAME_SHORT}_start"
END_FOLDER_NAME="avs_${MODEL_NAME_SHORT}_end"
INTERMEDIATE_FOLDER_NAME="avs_${MODEL_NAME_SHORT}_intermediate"
CONFIG_FOLDER="./avs_${MODEL_NAME_SHORT}_configs"
M2O_OUTPUT="avs_${MODEL_NAME_SHORT}_m2o"
M2M_OUTPUT="avs_${MODEL_NAME_SHORT}_m2m"

python3 ./generate_high_accuracy_model.py -n 1 -m "${MODEL_NAME}" -d cifar10 -p 1 -o "${START_FOLDER_NAME}"
python3 ./generate_high_accuracy_model.py -n 1 -m "${MODEL_NAME}" -d cifar10 -p 0 -o "${END_FOLDER_NAME}"
python3 ./find_high_accuracy_path_v2.py "./${START_FOLDER_NAME}/" to_vs --variance_sphere "./${END_FOLDER_NAME}/0.model.pt" -o "${M2O_OUTPUT}" -S --config "${CONFIG_FOLDER}/m2o.py"
mkdir "${INTERMEDIATE_FOLDER_NAME}"
cp ./avs_cct7_m2o/0-to_vs/1000.model.pt "./${INTERMEDIATE_FOLDER_NAME}/0.model.pt"
python3 ./find_high_accuracy_path_v2.py "./${INTERMEDIATE_FOLDER_NAME}" "./${END_FOLDER_NAME}" -o "${M2M_OUTPUT}" -S --config "${CONFIG_FOLDER}/m2m.py"