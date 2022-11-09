#!/bin/sh

# Slurm Directories
checkpoint_dir=/checkpoint/${USER}/${SLURM_JOB_ID}
ln -sfn $checkpoint_dir $PWD/checkpoint/${SLURM_JOB_ID}
touch $checkpoint_dir/DELAYPURGE

config_path=./configs/triplanar_nerf.yaml
data_dir=$HOME/workspace/data/nerf-format/

bash _experiments.sh $checkpoint_dir $data_dir
