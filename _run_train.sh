#!/bin/sh

# Slurm Directories
checkpoint_dir=/checkpoint/${USER}/${SLURM_JOB_ID}
ln -sfn $checkpoint_dir $PWD/checkpoint/${SLURM_JOB_ID}
touch $checkpoint_dir/DELAYPURGE

config_path=./configs/triplanar_nerf.yaml
data_dir=$HOME/workspace/data/nerf-format/

# File to keep track of rendered scenes because SLURM preemption
progress_file=$checkpoint_dir/rendered
touch $progress_file

for DATA_PATH in $(find $data_dir -type d -name "*train" -exec dirname "{}" \; | sort -u); do
    data_name=$(basename $DATA_PATH)

    triplanar_nerf_name="triplanar_nerf"-$data_name
    if ! grep -q $triplanar_nerf_name $progress_file; then
        echo $triplanar_nerf_name
        python3 app/main.py --config configs/triplanar_nerf.yaml --dataset-path $DATA_PATH --dataset-num-workers 4 --log-dir $checkpoint_dir --exp-name $triplanar_nerf_name
        echo $triplanar_nerf_name >> $progress_file
    fi

    # ngp_nerf_name="npg_nerf"-$data_name
    # if ! grep -q $ngp_nerf_name $progress_file; then
    #     echo $ngp_nerf_name
    #     python3 app/main.py --config configs/ngp_nerf.yaml --dataset-path $DATA_PATH --dataset-num-workers 4 --log-dir $checkpoint_dir --exp-name $ngp_nerf_name
    #     echo $ngp_nerf_name>> $progress_file
    # fi
done
