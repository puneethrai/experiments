OUTPUT_DIR=${1:-${OUTPUT:-"/mount_folder"}}

python classification_sample_async.py -m public\alexnet\FP32\alexnet.xml -i banana.jpg car.bmp -d GPU -p ${OUTPUT_DIR}