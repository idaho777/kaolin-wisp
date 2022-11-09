#!/bin/bash

# proxy_dir=$HOME/workspace/data/nerf-format/3d-proxy
# turna_dir=$HOME/workspace/data/nerf-format/turnarounds

#data_dir="$proxy_dir $turna_dir"
# data_dir="/data/joonho/workspace/data/nerf-format/mixamo/"
# checkpoint_dir=./_results/test/

config="./toon_template/toon_nerf.yaml"
bash _experiments.sh $config $checkpoint_dir $data_dir
