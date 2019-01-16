#!/bin/bash

INPUT_FILE=$1
OUTPUT_DIR=$2
RATIO=$3

SPLIT_RECORD_CNT=$(wc -l ${INPUT_FILE} | awk -v ratio=${RATIO} '{print int(ratio * $1)}')
shuf -o/tmp/$$.tmp ${INPUT_FILE}

sed "1,${SPLIT_RECORD_CNT}" /tmp/$$.tmp > ${OUTPUT_DIR}/train.data

((SPLIT_RECORD_CNT ++))
sed "${SPLIT_RECORD_CNT},$" /tmp/$$.tmp > ${OUTPUT_DIR}/eval.data
