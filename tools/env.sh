#!/bin/bash

#Modify these paths if required
export NUSCENES_PATH=/root/shared_data/nuscenes/v1.0-trainval
export MODELS_PATH=/root/shared_data/models
export SPLITS_FILE=/root/nuscenes-devkit/python-sdk/nuscenes/utils/splits.py

cp splits_valo.py $SPLITS_FILE

mkdir -p ../data/nuscenes
pushd ../data/nuscenes
ln -s $NUSCENES_PATH v1.0-trainval
popd

pushd ..
ln -s $MODELS_PATH
popd

. nusc_sh_utils.sh

#Optional
#export NSYS_PATH=/opt/nvidia/nsight-systems/2022.3.3/target-linux-tegra-armv8/nsys
#pushd /usr/bin
#ln -s $NSYS_PATH
#popd
