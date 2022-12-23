OUTPUT_DIR=${1:-${OUTPUT:-"/mount_folder"}}

curl -O https://storage.openvinotoolkit.org/data/test_data/images/banana.jpg
curl -O https://storage.openvinotoolkit.org/data/test_data/images/car.bmp

pip install -r requirements.txt
omz_downloader --name alexnet
omz_converter --name alexnet

python classification_sample_async.py -m public\alexnet\FP32\alexnet.xml -i banana.jpg car.bmp -d GPU -p ${OUTPUT_DIR}