#!/usr/bin/env bash

TASK_NAME="jddc"
MODEL_NAME="bert-base-chinese"
CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
export CUDA_VISIBLE_DEVICES="0"
export BERT_PRETRAINED_MODELS_DIR=$CURRENT_DIR/prev_trained_model/$MODEL_NAME
export DATA_DIR=$CURRENT_DIR/data


cd $DATA_DIR
if [ ! -d $TASK_NAME ]; then
  mkdir $TASK_NAME
  echo "makedir $DATA_DIR/$TASK_NAME"
fi

cd $TASK_NAME
#if [ ! -f "train.json" ] || [ ! -f "dev.json" ] || [ ! -f "test.json" ]; then
#  echo "data not exists"
#  exit 0
#else
#  echo "data exists"
#fi


# make output dir
if [ ! -d $CURRENT_DIR/${TASK_NAME}_output ]; then
  mkdir -p $CURRENT_DIR/${TASK_NAME}_output
  echo "makedir $CURRENT_DIR/${TASK_NAME}_output"
fi

# run task
cd $CURRENT_DIR
echo "Start running..."
if [ $# == 0 ]; then
    python3 run.py \
      --model_type=bert \
      --model_name_or_path=$BERT_PRETRAINED_MODELS_DIR \
      --task_name=$TASK_NAME \
      --do_train \
      --do_eval \
      --do_lower_case \
      --data_dir=$DATA_DIR/${TASK_NAME}/ \
      --max_seq_length=128 \
      --per_gpu_train_batch_size=64 \
      --per_gpu_eval_batch_size=64 \
      --learning_rate=2e-5 \
      --num_train_epochs=3.0 \
      --logging_steps=2146 \
      --save_steps=2146 \
      --output_dir=$CURRENT_DIR/${TASK_NAME}_output/ \
      --overwrite_output_dir \
      --seed=42
elif [ $1 == "predict" ]; then
    echo "Start predict..."
    python3 run.py \
      --model_type=bert \
      --model_name_or_path=$BERT_PRETRAINED_MODELS_DIR \
      --task_name=$TASK_NAME \
      --do_predict \
      --do_lower_case \
      --data_dir=$DATA_DIR/${TASK_NAME}/ \
      --max_seq_length=128 \
      --per_gpu_train_batch_size=64 \
      --per_gpu_eval_batch_size=64 \
      --learning_rate=2e-5 \
      --num_train_epochs=3.0 \
      --logging_steps=2146 \
      --save_steps=2146 \
      --output_dir=$CURRENT_DIR/${TASK_NAME}_output/ \
      --overwrite_output_dir \
      --seed=42
fi
