#!/bin/bash

# Modify the following depending on your platform
DEV_PLATFORM=jetson-xavier
#DEV_PLATFORM=jetson-orin

if [ "$DEV_PLATFORM" == "jetson-orin" ]; then
	CUDA_ARCH="8.7"
elif [ "$DEV_PLATFORM" == "jetson-xavier" ]; then
	CUDA_ARCH="7.2"
else
	printf "Please check https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/"
	printf "and set the CUDA_ARCH accordingly\n"
	CUDA_ARCH=""
fi

if [ "$CUDA_ARCH" != "" ]; then
	docker build . --build-arg CUDA_ARCH=$CUDA_ARCH -t valo:$DEV_PLATFORM
fi
