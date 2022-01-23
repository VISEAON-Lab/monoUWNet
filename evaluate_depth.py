from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from  torchvision.utils import save_image
from layers import disp_to_depth
from utils import readlines, sec_to_hm_str
from options import MonodepthOptions
import datasets
import networks
print(torch.__version__)
import matplotlib.pyplot as plt
from my_utils import *
import csv
from datetime import datetime 

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
def rank_error(errors, idx = 0, top = 5):
    list_err = []
    rank_list_maxi = []
    rank_list_mini = []
    errors = list(errors)
    for error in errors:
        list_err.append(list(error)[idx])
    copy_list_err = list_err.copy()
    list_err.sort(reverse=True)
    for value in list_err[:top]:
        rank_list_maxi.append(copy_list_err.index(value))
    print("maxi",rank_list_maxi)
    print(list_err[:top])
    for value in list_err[-top:]:
        rank_list_mini.append(copy_list_err.index(value))
    print("mini",rank_list_mini)
    print(list_err[-top:])
    return None
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    params = vars(opt)
      
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    device = torch.device("cpu" if opt.no_cuda else "cuda")

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"


    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    print("-> Loading weights from {}".format(opt.load_weights_folder))

    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "val_files.txt"))
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
    
    encoder_dict = torch.load(encoder_path, map_location=torch.device(device)) 
    # decoder_dict = torch.load(decoder_path) if torch.cuda.is_available() else torch.load(encoder_path,map_location = 'cpu')
    decoder_dict = torch.load(decoder_path, map_location=torch.device(device)) 
    # dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                        #    encoder_dict['height'], encoder_dict['width'],
                                        #    [0], 4, is_train=False)
    if opt.dataset=='sc': # aqualoc
        dataset = datasets.SCDataset(opt.dataset, opt.data_path, filenames,
                                        encoder_dict['height'], encoder_dict['width'],
                                        [0], 4, is_train=False, opts = opt) 
    elif opt.dataset=='uc': # aqualoc
        dataset = datasets.UCanyonDataset(opt.dataset, opt.data_path, filenames,
                                        encoder_dict['height'], encoder_dict['width'],
                                        [0], 4, is_train=False, opts = opt)

    dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)
     # TODO: check b4 the change if something hapened!!   
    encoder = networks.test_hr_encoder.hrnet18(False)
    encoder.num_ch_enc = [ 64, 18, 36, 72, 144 ]
    depth_decoder = networks.HRDepthDecoder(encoder.num_ch_enc, opt.scales)
    model_dict = encoder.state_dict()
    dec_model_dict = depth_decoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in dec_model_dict})
    
    encoder.to(device)
    encoder.eval()
    depth_decoder.to(device)
    depth_decoder.eval()
    
    pred_disps = []
    input_colors = []
    gt_depths = []
    print('-->Using\n cuda') if torch.cuda.is_available() else print('-->Using\n CPU')
    print("-> Computing predictions with size {}x{}".format(
        encoder_dict['width'], encoder_dict['height']))

    with torch.no_grad():
        # init_time = time.time()
        i = 0 
        for data in dataloader:
            i += 1  
            input_color = data[("color", 0, 0)].to(device)
            gt = data[("depth_gt")].to(device)   
            if opt.post_process:
                # Post-processed results require each image to have two forward passes
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

            output = depth_decoder(encoder(input_color))

            pred_disp_0, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
            pred_disp = pred_disp_0.cpu()[:, 0].numpy()
            #pred_disp_viz = pred_disp_0.squeeze()

            if opt.post_process:
                N = pred_disp.shape[0] // 2
                pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

            pred_disps.append(pred_disp)
            input_colors.append(toNumpy(input_color.cpu(), keepDim=True))
            gt_depths.append((gt.cpu()))

        # end_time = time.time()
        # inferring = end_time - init_time
        # print("===>total time:{}".format(sec_to_hm_str(inferring)))

    pred_disps = np.concatenate(pred_disps)
    input_colors = np.concatenate(input_colors)
    gt_depths = np.concatenate(gt_depths)
  
    
    
    save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions"+opt.model_name)
    print("-> Saving out benchmark predictions to {}".format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

 
    output_path = os.path.join(
            opt.load_weights_folder, "benchmark_predictions"+opt.model_name, "disps_{}_split.npy".format(opt.eval_split))
    print("-> Saving predicted disparities to ", output_path)
    np.save(output_path, pred_disps)




    print("-> Evaluating")

    errors = []
    ratios = []
    plt.set_cmap('jet')  
    tobe_cleaned = []
    cleaned = list(range(pred_disps.shape[0]))
    for i in tobe_cleaned:
        if i in cleaned:
            cleaned.remove(i)
    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_depth = np.squeeze(gt_depth)
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp
 
        mask = gt_depth > 0
        
        
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

        # dump images
        disp_resized = cv2.resize(pred_disps[i], (opt.width, opt.height))
        outPred = (normalize_numpy(pred_disps[i])*255).astype(np.uint8)
        inGT = (normalize_numpy(gt_depths[i])*255).astype(np.uint8)
        inGT = np.squeeze(inGT)
        inputColor = input_colors[i]
        # depth = 32.779243 / disp_resized
        # depth = np.clip(depth, 0, 80)/10
        # depth = np.uint8(depth * 256)
        save_path = os.path.join(save_dir, "{:010d}.png".format(i))
        plt.imsave(save_dir + "/frame_{:06d}_color.bmp".format(i), inputColor)
        plt.imsave(save_dir + "/frame_{:06d}_disp.bmp".format(i), outPred)
        plt.imsave(save_dir + "/frame_{:06d}_gt.bmp".format(i), inGT)


    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)
    ## ranked_error
    ranked_error = rank_error(errors, 0 ,10)
    
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")

    # log to file
    
    f = open('logger.csv','a')
    writer = csv.writer(f)
    headline=[" "] 
    writer.writerow(headline)
    headline=[opt.model_name]
    writer.writerow(headline)
    writer.writerow(params.keys())
    writer.writerow(params.values())

    time = str(datetime.now().isoformat(' ', 'seconds'))
    resdict = {
        "time": time,
        "abs_rel": np.round(mean_errors[0],3),
        "sq_rel": np.round(mean_errors[1],3),
        "rmse": np.round(mean_errors[2],3),
        "rmse_log": np.round(mean_errors[3],3),
        "a1": np.round(mean_errors[4],3),
        "a2": np.round(mean_errors[5],3),
        "a3": np.round(mean_errors[6],3),
        }
    writer.writerow(resdict.keys())
    writer.writerow(resdict.values())

    f.close()

if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
