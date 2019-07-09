#!/bin/bash

PTYOLOV3_HOME=/home/ubuntu/PyTorch-YOLOv3
MODEL_DIR=$1

rm -rf "$MODEL_DIR"/logs
rm -rf "$MODEL_DIR"/checkpoints
mkdir "$MODEL_DIR"/logs
mkdir "$MODEL_DIR"/checkpoints

python -m ptyolov3.train --model_def "$MODEL_DIR"/yolov3-tiny-custom.cfg \
                         --data_config "$MODEL_DIR"/custom.data \
                         --pretrained_weights "$PTYOLOV3_HOME"/weights/yolov3-tiny.weights \
                         --n_cpu 4 \
                         --epochs 71 \
                         --output_dir "$MODEL_DIR" \
                         --checkpoint_interval 5 \
                         --batch_size 10
                         /