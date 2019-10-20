#! /usr/bin/bash env

DATA_DIR="../../data/kinetics400/annotations"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} does not exist. Creating";
  mkdir -p ${DATA_DIR}
fi

wget https://storage.googleapis.com/deepmind-media/Datasets/kinetics_400.zip

unzip kinetics_400.zip
unzip -j kinetics_400_train.zip -d ${DATA_DIR}/
unzip -j kinetics_400_train.zip -d ${DATA_DIR}/
unzip -j kinetics_400_train.zip -d ${DATA_DIR}/

rm *.zip
rm -rf __MACOSX
