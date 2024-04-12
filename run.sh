#!/usr/bin/env bash

set +e
ulimit -n 4096

DATA_ROOT=./data
OUTPUT_ROOT=./output

DATASET=$1
if [ -z $DATASET ]; then
  echo "Usage: $0 <dataset> [gpu_id=0]"
  exit -1
fi

GPU_ID=${2:-'0'}
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo ">> GPU_ID: $GPU_ID"

EXTRA_ARGS=${@:2}
echo ">> EXTRA_ARGS: $EXTRA_ARGS"


split=false
case $DATASET in
  t|train)
    source_path=$DATA_ROOT/tandt/train
    model_path=$OUTPUT_ROOT/train
    ;;
  l|lego)
    source_path=$DATA_ROOT/nerf_synthetic/lego
    model_path=$OUTPUT_ROOT/lego
    ;;
  g|gate)
    source_path=$DATA_ROOT/phototourist/brandenburg_gate
    model_path=$OUTPUT_ROOT/brandenburg_gate
    split=true
    ;;
  f|fountain)
    source_path=$DATA_ROOT/phototourist/trevi_fountain
    model_path=$OUTPUT_ROOT/trevi_fountain
    split=true
    ;;
  *)
    echo ">> Error: unknown dataset \"$DATASET\""
    exit 1
    ;;
esac
echo ">> DATASET: $(basename $model_path)"


echo "[Step 1/3] train"
if [ $split == true ]; then
  python train.py --eval -s $source_path -m $model_path $EXTRA_ARGS
else
  python train.py -s $source_path -m $model_path $EXTRA_ARGS
fi

echo "[Step 2/3] render"
python render.py -m $model_path $EXTRA_ARGS

echo "[Step 3/3] metrics"
python metrics.py -m $model_path $EXTRA_ARGS
