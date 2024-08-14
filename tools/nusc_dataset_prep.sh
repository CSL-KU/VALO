#!/bin/bash

. nusc_sh_utils.sh

rm -rf token_to_*.json
nusc_revert_tables

export CALIBRATION=${CALIBRATION:-0}
export DATASET_PERIOD=${DATASET_PERIOD:-500}
#export DATASET_RANGE=${DATASET_RANGE:-"48-53"}

period=$DATASET_PERIOD
TABLES_PATH="nusc_tables_and_dicts/$period"
rm -rf $TABLES_PATH
mkdir -p "$TABLES_PATH/tables"

python nusc_dataset_utils.py populate_annos_v2 50
mv -f sample.json sample_data.json instance.json ego_pose.json \
	sample_annotation.json scene.json "$TABLES_PATH/tables"
nusc_link_tables "$TABLES_PATH/tables"
if [ $period != 50 ]; then
	python nusc_dataset_utils.py prune_annos $period
	mv -f sample.json sample_data.json instance.json  ego_pose.json \
		sample_annotation.json scene.json "$TABLES_PATH/tables"
fi

python nusc_dataset_utils.py generate_dicts
mv -f token_to_anns.json token_to_pos.json $TABLES_PATH

# now generate data as well
gen_data
mkdir -p "$TABLES_PATH/generated_data"
save_data "$TABLES_PATH/generated_data"

nusc_revert_tables

unset CALIBRATION DATASET_PERIOD
