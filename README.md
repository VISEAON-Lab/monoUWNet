# Self-Supervised Monocular Depth Underwater
This repo is for Self-Supervised Monocular Depth Underwater paper which can be found here:
https://arxiv.org/abs/2210.03206

The work is mostly based on DiffNet which can be found here:
https://github.com/brandleyzhou/DIFFNet


# requirements:
matplotlib==3.4.2
numpy==1.21.2
opencv-python==4.5.2.52
Pillow==8.4.0
scikit-image==0.18.3
scipy==1.7.1
tensorboard==2.7.0
tensorboardX==2.4
torch==1.10.1
torchvision==0.2.1


Running training over all 4 FLC datasets together and evaluating on each one seperatly (ar all together):


#!/bin/bash
echo sleeping..
sleep 18000
dname=FLC_4DS_tiny_sky
date=20220706
ds=FLC_4DS_tiny_sky
##################################################3
# train

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


# evaluate - option 1: run on each sub ds by it self
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

# evaluate - option 2: run on the unified ds:
echo "running flc_new evaluation on ${ds}"
python evaluate_depth.py \
--model_name="${run_name}_eval_${ds}" \
--dataset=uc \
--eval_mono \
--load_weights_folder="/home/samitai/Work/myDIFFNet/models/${run_name}/models/weights_last" \
--data_path /home/samitai/Work/Datasets/ANSFL/allData2 \
--save_pred_disps \
--use_depth \
--eval_split=${ds} \
--eval_sky 

