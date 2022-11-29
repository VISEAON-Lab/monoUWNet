
#!/bin/bash
echo sleeping..
sleep 18000
dname=FLC_4DS_tiny_sky
date=20220706

##################################################3
# # 1) basic

echo "train diffnet flc2 pytorch"
run_name=${date}_FLC_${dname}

# echo "${run_name}_test"
python train.py --png \
--model_name=$run_name \
--data_path="/home/samitai/Work/Datasets/ANSFL/allData2" \
--dataset="uc" \
--split=${dname} \
--height=480 \
--width=640 \
--batch_size=8 \
--num_epochs=20 \
--load_weights_folder="/home/samitai/Work/myDIFFNet/models/diffnet_1024x320" \
--do_flip \
# # # # # --use_corrLoss \
# # # # # --use_lvw \
# # # # --use_recons_net

for ds in uc flatiron tiny
do
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
done

# ############################################
# 2) corrLoss

echo "train diffnet flc2 pytorch"
run_name=${date}_FLC_corrLoss_${dname}

echo "${run_name}_test"
python train.py --png \
--model_name=$run_name \
--data_path="/home/samitai/Work/Datasets/ANSFL/allData2" \
--dataset="uc" \
--split=${dname} \
--height=480 \
--width=640 \
--batch_size=8 \
--num_epochs=20 \
--load_weights_folder="/home/samitai/Work/myDIFFNet/models/diffnet_1024x320" \
--do_flip \
--use_corrLoss 


for ds in uc flatiron tiny
do
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
done

############################################
# 3) lvw
echo "train diffnet flc2 pytorch"
run_name=${date}_FLC_lvw_${dname}

echo "${run_name}_test"
python train.py --png \
--model_name=$run_name \
--data_path="/home/samitai/Work/Datasets/ANSFL/allData2" \
--dataset="uc" \
--split=${dname} \
--height=480 \
--width=640 \
--batch_size=8 \
--num_epochs=20 \
--load_weights_folder="/home/samitai/Work/myDIFFNet/models/diffnet_1024x320" \
--do_flip \
--use_lvw

for ds in uc flatiron tiny
do
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
done

############################################

# 4) rhf
echo "train diffnet flc2 pytorch"
run_name=${date}_FLC_rhf_${dname}

echo "${run_name}_test"
python train.py --png \
--model_name=$run_name \
--data_path="/home/samitai/Work/Datasets/ANSFL/allData2" \
--dataset="uc" \
--split=${dname} \
--height=480 \
--width=640 \
--batch_size=8 \
--num_epochs=20 \
--load_weights_folder="/home/samitai/Work/myDIFFNet/models/diffnet_1024x320" \
--do_flip \
--use_homomorphic_filt

for ds in uc flatiron tiny
do
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
done

############################################
# 5) all
echo "train diffnet flc2 pytorch"
run_name=${date}_FLC_all_${dname}

echo "${run_name}_test"
python train.py --png \
--model_name=$run_name \
--data_path="/home/samitai/Work/Datasets/ANSFL/allData2" \
--dataset="uc" \
--split=${dname} \
--height=480 \
--width=640 \
--batch_size=8 \
--num_epochs=20 \
--load_weights_folder="/home/samitai/Work/myDIFFNet/models/diffnet_1024x320" \
--do_flip \
--use_corrLoss \
--use_lvw \
--use_homomorphic_filt

for ds in uc flatiron tiny
do
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
done

############################################
# 6) all-corrLoss
echo "train diffnet flc2 pytorch"
run_name=${date}_FLC_all_wo_corrLoss_${dname}

echo "${run_name}_test"
python train.py --png \
--model_name=$run_name \
--data_path="/home/samitai/Work/Datasets/ANSFL/allData2" \
--dataset="uc" \
--split=${dname} \
--height=480 \
--width=640 \
--batch_size=8 \
--num_epochs=20 \
--load_weights_folder="/home/samitai/Work/myDIFFNet/models/diffnet_1024x320" \
--do_flip \
--use_lvw \
--use_homomorphic_filt

for ds in uc flatiron tiny
do
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
done


#######################33

# 7) all-lvw
echo "train diffnet flc2 pytorch"
run_name=${date}_FLC_all_wo_lvw_${dname}

echo "${run_name}_test"
python train.py --png \
--model_name=$run_name \
--data_path="/home/samitai/Work/Datasets/ANSFL/allData2" \
--dataset="uc" \
--split=${dname} \
--height=480 \
--width=640 \
--batch_size=8 \
--num_epochs=20 \
--load_weights_folder="/home/samitai/Work/myDIFFNet/models/diffnet_1024x320" \
--do_flip \
--use_corrLoss \
--use_homomorphic_filt

for ds in uc flatiron tiny
do
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
done


###########################
## 8) all-hmf
# echo "train diffnet flc2 pytorch"
# run_name=${date}_FLC_all_wo_rhf_${dname}

# echo "${run_name}_test"
# python train.py --png \
# --model_name=$run_name \
# --data_path="/home/samitai/Work/Datasets/ANSFL/allData2" \
# --dataset="uc" \
# --split=${dname} \
# --height=480 \
# --width=640 \
# --batch_size=8 \
# --num_epochs=20 \
# --load_weights_folder="/home/samitai/Work/myDIFFNet/models/diffnet_1024x320" \
# --do_flip \
# --use_corrLoss \
# --use_lvw

# for ds in uc flatiron tiny
# do
#     echo "running flc_new evaluation on ${ds}"
#     python evaluate_depth.py \
#     --model_name="${run_name}_eval_${ds}" \
#     --dataset=uc \
#     --eval_mono \
#     --load_weights_folder="/home/samitai/Work/myDIFFNet/models/${run_name}/models/weights_last" \
#     --data_path /home/samitai/Work/Datasets/ANSFL/allData2 \
#     --save_pred_disps \
#     --use_depth \
#     --eval_split=${ds} \
#     --eval_sky 
# done
