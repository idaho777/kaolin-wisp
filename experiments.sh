
data_nerf_dir=$HOME/workspace/data/nerf-format/3d-proxy
data_turn_dir=$HOME/workspace/data/nerf-format/turnarounds
# data_dir=$HOME/workspace/data/nerf-format/mini-test
checkpoint_dir=./_results/test/

mkdir -p $checkpoint_dir

# File to keep track of rendered scenes because SLURM preemption
progress_file=$checkpoint_dir/rendered
touch $progress_file

for DATA_PATH in $(find $data_nerf_dir $data_turn_dir -type d -name "*train" -exec dirname "{}" \; | sort -u); do
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
    #     # python3 app/main.py --config configs/ngp_nerf.yaml --dataset-path $DATA_PATH --dataset-num-workers 4 --log-dir $checkpoint_dir --exp-name $ngp_nerf_name
    #     echo $ngp_nerf_name>> $progress_file
    # fi
done
