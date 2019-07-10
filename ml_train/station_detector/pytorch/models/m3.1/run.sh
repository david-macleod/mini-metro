#!/bin/bash

PTYOLOV3_HOME=$1
MODEL_DIR="$(dirname "$0")"

rm -rf "$MODEL_DIR"/logs
rm -rf "$MODEL_DIR"/checkpoints
mkdir "$MODEL_DIR"/logs
mkdir "$MODEL_DIR"/checkpoints

# multiscale_training hack as argparse has incorrect implementation for boolean
python -m ptyolov3.train --model_def "$MODEL_DIR"/yolov3-custom.cfg \
                         --data_config "$MODEL_DIR"/custom.data \
                         --pretrained_weights "$PTYOLOV3_HOME"/weights/yolov3-tiny.weights \
                         --n_cpu 4 \
                         --epochs 71 \
                         --output_dir "$MODEL_DIR" \
                         --checkpoint_interval 10 \
                         --batch_size 10 \
                         --img_size 608 \
                         --multiscale_training ''
                         /