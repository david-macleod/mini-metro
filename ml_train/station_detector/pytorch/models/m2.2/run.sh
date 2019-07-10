#!/bin/bash

PTYOLOV3_HOME=$1
MODEL_DIR="$(dirname "$0")"

rm -rf "$MODEL_DIR"/logs
rm -rf "$MODEL_DIR"/checkpoints
mkdir "$MODEL_DIR"/logs
mkdir "$MODEL_DIR"/checkpoints

python -m ptyolov3.train --model_def "$MODEL_DIR"/yolov3-custom.cfg \
                         --data_config "$MODEL_DIR"/custom.data \
                         --pretrained_weights "$PTYOLOV3_HOME"/weights/darknet53.conv.74 \
                         --n_cpu 4 \
                         --epochs 41 \
                         --output_dir "$MODEL_DIR" \
                         --checkpoint_interval 5 \
                         --batch_size 10
                         /