#! /bin/sh

curl -O https://storage.openvinotoolkit.org/data/test_data/images/banana.jpg
curl -O https://storage.openvinotoolkit.org/data/test_data/images/car.bmp

pip install -r requirements.txt
omz_downloader --name alexnet
omz_converter --name alexnet