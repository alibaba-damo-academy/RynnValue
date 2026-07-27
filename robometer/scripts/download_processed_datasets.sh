#!/bin/bash
# check ROBOMETER_PROCESSED_DATASETS_PATH (or RBM_PROCESSED_DATASETS_PATH) is set
if [ -z "${ROBOMETER_PROCESSED_DATASETS_PATH:-$RBM_PROCESSED_DATASETS_PATH}" ]; then
    echo "ROBOMETER_PROCESSED_DATASETS_PATH (or RBM_PROCESSED_DATASETS_PATH) is not set"
    exit 1
fi
# increase the number of open files limit
ulimit -n 65535
# download processed datasets
hf download jesbu1/processed_datasets --repo-type dataset --local-dir=${ROBOMETER_PROCESSED_DATASETS_PATH:-$RBM_PROCESSED_DATASETS_PATH}