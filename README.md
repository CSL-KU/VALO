# VALO
A Versatile Anytime Framework for LiDAR based Object Detection Deep Neural Networks (EMSOFT 2024)

This project is built on top of OpenPCDet (https://github.com/open-mmlab/OpenPCDet). We thank them for their wonderful work.

# HOW TO install and run the project to reproduce the results published in the paper

Here we provide how to configure NVIDIA Jetson AGX Xavier to run VALO with pretrained models. Steps to install VALO on x86 systems to train the DNN models will also be available soon. 
Before you start, make sure the L4T version of your Jetson is at least R35.2.1. You can check its version from /etc/nv_tegra_release file.

Configure the Jetson as follows:
```
sudo sysctl -w kernel.sched_rt_runtime_us=-1
sudo jetson_clocks
sudo jetson_clocks --fan
sudo bash -c "echo 'performance' > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"
```
Create a directory to be shared with the docker container (e.g. shared_data).
Download the pretrained models from  [here](https://kansas-my.sharepoint.com/:u:/g/personal/a249s197_home_ku_edu/EdnqLdheA8FJhE6SwRi1_TwBAy6Z9wMpw697_EIGcbHT5w?e=Gu69RC) and nuScenes dataset from [nuscenes.org](https://www.nuscenes.org/nuscenes#download). Extract the models and the dataset into the shared directory with the following hierarchy:
```
shared_data/nuscenes/v1.0-trainval/
|-- maps
|-- samples
|-- sweeps
`-- v1.0-trainval

shared_data/models/
|-- cbgs_voxel0075_centerpoint_5swipes.pth
|-- cbgs_voxel0075_res3d_centerpoint_anytime_18.pth
|-- cbgs_voxel0075_res3d_centerpoint_anytime_v1.pth
|-- cbgs_voxel01_centerpoint_5swipes.pth
|-- cbgs_voxel01_res3d_centerpoint_anytime_16.pth
|-- cbgs_voxel02_centerpoint_5swipes.pth
`-- voxelnext_nuscenes_kernel1.pth
```

Change directory to VALO/docker then modify docker_build.sh to set the development platform if needed. Then source the script to build the docker image:
```
. docker_build.sh
```

Set the shared_data path. Example:
```
SHARED_DIR=/home/nvidia/ssd/shared_data
```

Run the image:
```
docker run --runtime nvidia --net host -it --privileged --cap-add=ALL  --ulimit rtprio=99 -v $SHARED_DIR:/root/shared_data --name valo_container valo:$DEV_PLATFORM
```

Run the following commands to setup:
```
cd ~/VALO/tools
. env.sh
prune_train_data_from_tables # only do if not going to train
```

Finally, run the calib and test script to do the calibration and generate the results:
```
. calib_and_run_tests.sh
for i in $(seq 0 3)
do
  python log_plotter.py exp_data_nsc_methods/ $i
done
```
The plotted results of the experiments will be available in shared_data/exp_plots .
	
	


