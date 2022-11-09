# bash _experiments_same.sh ./toon_template/toon_nerf_sum.yaml ./_results/cat-sum/ /data/joonho/workspace/data/nerf-format/mini-test/amy-studio0-0064/
# bash _experiments_same.sh ./toon_template/toon_nerf_sum.yaml ./_results/cat-sum/ /data/joonho/workspace/data/nerf-format/mini-test/amy-studio0-crop-0032/
# bash _experiments_same.sh ./toon_template/toon_nerf_sum.yaml ./_results/cat-sum/ /data/joonho/workspace/data/nerf-format/mini-test/kachujin-studio0-crop-0032
# bash _experiments_same.sh ./toon_template/toon_nerf_sum.yaml ./_results/cat-sum/ /data/joonho/workspace/data/nerf-format/mini-test/timmy-studio0-crop-0032/
# bash _experiments_same.sh ./toon_template/toon_nerf_sum.yaml ./_results/cat-sum/ /data/joonho/workspace/data/nerf-format/mini-test/jackie-studio0-crop-0032/

# bash _experiments_same.sh ./toon_template/toon_nerf_sum.yaml ./_results/cat-sum/ /data/joonho/workspace/data/nerf-format/mini-test/planes-0064/
# bash _experiments_same.sh ./toon_template/toon_nerf_sum.yaml ./_results/cat-sum/ /data/joonho/workspace/data/nerf-format/mini-test/shape-0064/
# bash _experiments_same.sh ./toon_template/toon_nerf_sum.yaml ./_results/cat-sum/ /data/joonho/workspace/data/nerf-format/mini-test/lego-0064/
# bash _experiments_same.sh ./toon_template/toon_nerf_sum.yaml ./_results/cat-sum/ /data/joonho/workspace/data/nerf-format/mini-test/lego-flat-color-0064/
# bash _experiments_same.sh ./toon_template/toon_nerf_sum.yaml ./_results/cat-sum/ /data/joonho/workspace/data/nerf-format/mini-test/lego-studio-color-0064/


# for d in $(ls -d /data/joonho/workspace/data/nerf-format/mini-test/*); do
#     bash _experiments_same.sh ./toon_template/toon_nerf_sum.yaml ./_results/2022-10-31-full-image-repeated $d
# done

date=$(date '+%Y-%m-%d')
bash _experiments.sh ./toon_template/toon_nerf_sum_adam.yaml ./_results/$date-full-run-adam /data/joonho/workspace/data/nerf-format/base-test/