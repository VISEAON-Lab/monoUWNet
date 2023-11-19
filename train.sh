#!/bin/bash
dname=FLC_4DS_tiny_sky
date=20220706
ds=FLC_4DS_tiny_sky

echo "train diffnet flc2 pytorch"
run_name=${date}FLC${dname}

echo "${run_name}_test"
# python train.py --png --model_name=$run_name --data_path="/workspaces/monoUWNet/datasets/allData2" --dataset="uc" --split=${dname} --height=480 --width=640 --batch_size=8 --num_epochs=20 --load_weights_folder= --do_flip --use_corrLoss --use_lvw --use_recons_net
python train.py \
    --png \
    --model_name=$run_name \
    --data_path="/workspaces/monoUWNet/datasets/allData2" \
    --dataset="uc" \
    --split=${dname} \
    --height=480 \
    --width=640 \
    --batch_size=4 \
    --num_epochs=20 \
    --do_flip \
    --use_corrLoss \
    --use_lvw \
    --use_recons_net
