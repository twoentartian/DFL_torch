#!/bin/bash

MODEL_NAME="cct_7_3x1_32"
DATASET_NAME="cifar100"
CONFIG="./configs/densenet.py"
NUMBER_OF_MODELS="1"
START_INDEX="0"
WORKER_COUNT="1"
CORE_COUNT="16"


START_FOLDER="${MODEL_NAME}_${DATASET_NAME}_start_${NUMBER_OF_MODELS}"
END_FOLDER="${MODEL_NAME}_${DATASET_NAME}_start_${NUMBER_OF_MODELS}"

python3 ./generate_high_accuracy_model.py -n "${NUMBER_OF_MODELS}" -c "${CORE_COUNT}" -w "${WORKER_COUNT}" -i "${START_INDEX}" -m "${MODEL_NAME}" -d "${DATASET_NAME}" -o "${START_FOLDER}"
python3 ./generate_high_accuracy_model.py -n "${NUMBER_OF_MODELS}" -c "${CORE_COUNT}" -w "${WORKER_COUNT}" -i "${START_INDEX}" -m "${MODEL_NAME}" -d "${DATASET_NAME}" -o "${END_FOLDER}"
python3 ./find_high_accuracy_path_v2.py "${START_FOLDER}" "${END_FOLDER}" --config "${CONFIG}" -S -w "${WORKER_COUNT}" -c "${CORE_COUNT}" -o "m2m_${MODEL_NAME}_${DATASET_NAME}" --amp