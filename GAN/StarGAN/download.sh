#!/bin/bash

echo "I referred to https://github.com/yunjey/stargan/blob/master/download.sh"

# CelebA images and attribute labels
URL=https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=0
mkdir -p ./dataset/
ZIP_FILE=./dataset/celeba.zip
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d ./dataset/
rm ./dataset/celeba.zip

