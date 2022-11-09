# This script will run Kaolin-wisp on all datasets in data_dir using the config_yaml file as hyperparameters and save the 
#   results in a restuls_dir.  We go through each dataset, run kaolin-wisp and save the values.  This script has basic 
#   checkpointing by saving completed dataset names in a progress file.  The progress file must have the same number of
#   lines as number of data_dirs_count in order to start a new run, otherwise the script will think the experiment was
#   not completed.
#
# How to call: 
#   - bash experiments.sh <config_yaml> <results_dir> <data_dir>
# config_yaml: configuration file for nerf runs
# data_dir: list of directories with nerf-formatted data
# results_dir: directory with all results.
#   Note that this contains experiment directories with date times


### Data directory to Nerfify
echo "running experiments.sh"
input_dir=${@:3}
data_dir_cmd="find $input_dir -type d -name "*train" -exec dirname "{}" \; | sort -u"
# data_dirs_count=$(eval "$data_dir_cmd" | wc -l)
data_dirs_count=1
data_dirs=$(eval "$data_dir_cmd" | shuf | head -n $data_dirs_count)


### Resume unfinished renders or start a new one.
# Create output directory
progress_file_name="rendered"
# results_dir="./_results/test/"
results_dir=$2
mkdir -p $results_dir

# Get existing progress file if possible
checkpoint_dir=$(ls -td $results_dir | head -1)
progress_file=$checkpoint_dir/$progress_file_name

# Check if progress file exists and has not ran through all data
if [ -f "$progress_file" ] && [ ! "$(grep -e '<done>' $progress_file)" ]; then
    echo "Found incomplete run.  <done> was not found."
    echo "Resume $checkpoint_dir"
else
    # Create new checkpoint_dir, progress file, and copy config
    date_now=$(date "+%F-%H-%M-%S")
    echo "Start new run $date_now"
    checkpoint_dir=$results_dir/$date_now
    mkdir -p $checkpoint_dir

    # Create fresh progress file
    progress_file=$checkpoint_dir/$progress_file_name
    touch $progress_file
fi

# Copy running script to checkpoint_dir
cp $0 $checkpoint_dir

# Copy config file to checkpoint_dir
config_path=$1
experiment_name=$(basename $config_path .yaml)
cp $config_path $checkpoint_dir

# Create for notes
readme_path=$checkpoint_dir/README
touch $readme_path
echo $@ > $readme_path


gifs_dir=$checkpoint_dir/gifs
mkdir -p $gifs_dir

# File to keep track of rendered scenes because SLURM preemption
i=1
while [[ $i -le 5 ]]; do
    DATA_PATH=$data_dirs
    data_name=$(basename $DATA_PATH)

    triplanar_nerf_name=$experiment_name-$data_name-$i
    # Check if progress file contains the current dataset.
    if ! grep -q $triplanar_nerf_name $progress_file; then
        echo $triplanar_nerf_name
        WISP_HEADLESS=1 python3 ./toon_template/toon_main.py --config $config_path --dataset-path $DATA_PATH --dataset-num-workers 4 --log-dir $checkpoint_dir --exp-name $triplanar_nerf_name
        # Log completed dataset into progress file
        echo $triplanar_nerf_name >> $progress_file

        # bash /data/joonho/workspace/output/kaolin-to-gif.sh $checkpoint_dir $gifs_dir
    fi
    ((i++))
done
echo "<done>" >> $progress_file

gifs_dir=$checkpoint_dir/gifs
mkdir -p $gifs_dir
bash /data/joonho/workspace/output/kaolin-to-gif.sh $checkpoint_dir $gifs_dir