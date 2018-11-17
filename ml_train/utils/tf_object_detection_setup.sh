pretrained_model=ssd_mobilenet_v1_coco_2018_01_28

source activate tensorflow

# install dependencies
sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
pip install Cython
pip install jupyter
pip install matplotlib
conda install jpeg

git clone https://github.com/david-macleod/mini-metro

# clone object detection repo (with python 3 fixes)
git clone -b dev https://github.com/david-macleod/models.git

# clone cocoapi and run make
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools ~/models/research

cd ~/models/research

# add research/slim directions to PYTHONPATH
echo "export PYTHONPATH=$PYTHONPATH:~/models/research:~/models/research/slim" >> ~/.bashrc

# compile protos
wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
unzip protobuf.zip
./bin/protoc object_detection/protos/*.proto --python_out=.

# run test
python object_detection/builders/model_builder_test.py

# get pre-trained model
cd ~
wget http://download.tensorflow.org/models/object_detection/${pretrained_model}.tar.gz
tar -xvf ${pretrained_model}.tar.gz
mkdir -p model_zoo/${pretrained_model}
cp ${pretrained_model}/model.ckpt.* model_zoo/${pretrained_model}/

# clean up
rm -rf ${pretrained_model}*

# create train output directory
mkdir tf_data