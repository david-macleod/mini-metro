MODEL=$1
CONFIG=$2
PIPELINE_CONFIG_PATH="models/${MODEL}/${CONFIG}"

for i in {1..5}
do
  MODEL_DIR="/home/ubuntu/tf_data/station_detector/${MODEL}/train${i}"
  python -m object_detection.model_main --model_dir=$MODEL_DIR --pipeline_config_path=$PIPELINE_CONFIG_PATH
done