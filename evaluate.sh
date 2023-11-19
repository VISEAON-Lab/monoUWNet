#!/bin/bash
dname=FLC_4DS_tiny_sky
date=20220706
ds=FLC_4DS_tiny_sky

echo "train diffnet flc2 pytorch"
run_name=${date}FLC${dname}

# echo "${run_name}_test"
# # python train.py --png --model_name=$run_name --data_path="/workspaces/monoUWNet/datasets/allData2" --dataset="uc" --split=${dname} --height=480 --width=640 --batch_size=8 --num_epochs=20 --load_weights_folder= --do_flip --use_corrLoss --use_lvw --use_recons_net
# python train.py --png --model_name=$run_name --data_path="/workspaces/myDIFFNet/datasets/allData2" --dataset="uc" --split=${dname} --height=480 --width=640 --batch_size=4 --num_epochs=20 --do_flip --use_corrLoss --use_lvw --use_recons_net


echo "running flc_new evaluation on ${ds}"
python evaluate_depth.py \
--model_name="${run_name}eval${ds}" \
--dataset=uc \
--eval_mono \
--load_weights_folder="/workspaces/monoUWNet/weights_last" \
--data_path="/workspaces/monoUWNet/datasets/allData2" \
--save_pred_disps \
--use_depth \
--eval_split=${ds} \
--eval_sky