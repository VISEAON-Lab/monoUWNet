# Self-Supervised Monocular Depth Underwater
This repo is for Self-Supervised Monocular Depth Underwater paper which can be found here:
https://arxiv.org/abs/2210.03206

The work is mostly based on DiffNet which can be found here:
https://github.com/brandleyzhou/DIFFNet




Running training over all 4 FLC datasets together and evaluating on each one seperatly:


#!/bin/bash
echo sleeping..
sleep 18000
dname=FLC_4DS_tiny_sky
date=20220706

##################################################3
1) basic

echo "train diffnet flc2 pytorch"
run_name=${date}_FLC_${dname}

echo "${run_name}_test"
python train.py --png \
--model_name=$run_name \
--data_path=<datapath> \
--dataset="uc" \
--split=${dname} \
--height=480 \
--width=640 \
--batch_size=8 \
--num_epochs=20 \
--load_weights_folder=<initial weights Folder> \
--do_flip \
--use_corrLoss \
--use_lvw \
--use_recons_net

for ds in uc flatiron tiny
do
    echo "running flc_new evaluation on ${ds}"
    python evaluate_depth.py \
    --model_name="${run_name}_eval_${ds}" \
    --dataset=uc \
    --eval_mono \
    --load_weights_folder=<weightsFolder> \
    --data_path <datapath> \
    --save_pred_disps \
    --use_depth \
    --eval_split=${ds} \
    --eval_sky 
done

