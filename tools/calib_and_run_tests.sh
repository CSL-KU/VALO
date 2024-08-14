#!/bin/bash
. nusc_sh_utils.sh

# CALIBRATION, only needs to be done once to generate calib_data files
export CALIBRATION=1
export DATASET_PERIOD="500" # milliseconds
./nusc_dataset_prep.sh
link_data 500
export CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_voxel0075_res3d_centerpoint_anytime_18.yaml"
export CKPT_FILE="../models/cbgs_voxel0075_res3d_centerpoint_anytime_18.pth"
DEADLINE_SEC=100.0 # ignore deadline
./run_tests.sh singlem 4 $DEADLINE_SEC
./run_tests.sh singlem 5 $DEADLINE_SEC
ln -s calib_data_m4_c18.json calib_data_m8_c18.json
ln -s calib_data_m4_c18.json calib_data_m9_c18.json

export CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_voxel0075_voxelnext_anytime.yaml"
export CKPT_FILE="../models/voxelnext_nuscenes_kernel1.pth"
./run_tests.sh singlem 6 $DEADLINE_SEC
./run_tests.sh singlem 7 $DEADLINE_SEC

export CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_voxel01_res3d_centerpoint_anytime_16.yaml"
export CKPT_FILE="../models/cbgs_voxel01_res3d_centerpoint_anytime_16.pth"
./run_tests.sh singlem 11 $DEADLINE_SEC

export CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_voxel0075_res3d_centerpoint_anytime_v1.yaml"
export CKPT_FILE="../models/cbgs_voxel0075_res3d_centerpoint_anytime_v1.pth"
export TASKSET="taskset 0x3f"
export OMP_NUM_THREADS=2
export USE_ALV1=1
./run_tests.sh singlem $DEADLINE_SEC
unset OMP_NUM_THREADS USE_ALV1 TASKSET CFG_FILE CKPT_FILE DEADLINE_RANGE_MS DATASET_RANGE

# TESTING
export CALIBRATION=0
export DATASET_PERIOD="350"
./nusc_dataset_prep.sh
link_data 350 
./run_tests.sh methods 0.090 0.065 0.350
python eval_from_files.py ./exp_data_nsc_methods
