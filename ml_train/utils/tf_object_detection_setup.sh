pretrained_model=ssd_mobilenet_v1_coco_2018_01_28

# install dependencies
apt-get install protobuf-compiler python-pil python-lxml python-tk
pip install Cython
pip install jupyter
pip install matplotlib

# clone object detection repo
git clone https://github.com/tensorflow/models.git

# clone cocoapi and run make
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI; make; cp -r pycocotools /content/models/research/

cd /content/models/research

# add research/slim directions to PYTHONPATH
%set_env PYTHONPATH=/content/models/research:/content/models/research/slim

# compile protos
protoc object_detection/protos/*.proto --python_out=.

# run test
python object_detection/builders/model_builder_test.py

# get pre-trained model
wget http://download.tensorflow.org/models/object_detection/${pretrained_model}.tar.gz
tar -xvf ${pretrained_model}.tar.gz
cp ${pretrained_model}/model.ckpt.* .