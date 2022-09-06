from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from  torchvision.utils import save_image
import torch.nn.functional as fn
from layers import disp_to_depth
from utils import readlines, sec_to_hm_str, estimateA, water_types_Nrer_rgb
from options import MonodepthOptions
import datasets
import networks
print(torch.__version__)
import matplotlib.pyplot as plt
from my_utils import *
import csv
from datetime import datetime 
from skyPixelSegmentation import find_sky_mask
from layers import CorrelationLoss
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

    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
    
    encoder_dict = torch.load(encoder_path, map_location=torch.device(device)) 
    # decoder_dict = torch.load(decoder_path) if torch.cuda.is_available() else torch.load(encoder_path,map_location = 'cpu')
    decoder_dict = torch.load(decoder_path, map_location=torch.device(device)) 
    if  opt.dataset=='kitti':
        dataset = datasets.KITTIRAWDataset(opt.dataset, opt.data_path, filenames,
                                           encoder_dict['height'], encoder_dict['width'],
                                           [0], 4, is_train=False, opts = opt)
    elif opt.dataset=='sc': # aqualoc
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
    
    if opt.use_recons_net:
        reconNet_path = os.path.join(opt.load_weights_folder, "recon.pth")
        reconNet_dict = torch.load(reconNet_path, map_location=torch.device(device)) 
        reconNet = networks.WaterTypeRegression(3)
        recon_model_dict = reconNet.state_dict()
        reconNet.load_state_dict({k: v for k, v in reconNet_dict.items() if k in recon_model_dict})
        
        reconNet.to(device)
        reconNet.eval()

    pred_disps = []
    pred_Js = []
    input_colors = []
    gt_depths = []
    sky_masks=[]
    print('-->Using\n cuda') if torch.cuda.is_available() else print('-->Using\n CPU')
    print("-> Computing predictions with size {}x{}".format(
        encoder_dict['width'], encoder_dict['height']))
    
    save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions"+opt.model_name)
    print("-> Saving out benchmark predictions to {}".format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    saveFig=True
    if saveFig:
        fig = plt.figure()

    # corr_loss = CorrelationLoss()
    tot_corr_gt2pred=0
    tot_corr_gt2ulap=0
    tot_corr_pred2ulap=0
    with torch.no_grad():
        # init_time = time.time()
        i = 0 
        for data in dataloader:
            i += 1  
            input_color = data[("color", 0, 0)].to(device)
            gt = data[("depth_gt")].to(device)   
            if torch.count_nonzero(gt) == 0:
                continue
            if opt.post_process:
                # Post-processed results require each image to have two forward passes
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

            output = depth_decoder(encoder(input_color))

            pred_disp_0, depth = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)

            if opt.use_recons_net:
                J, coeffs = reconNet(input_color.to(device),depth.to(device) )
                pred_Js.append(normalize_numpy(toNumpy(J.cpu(), keepDim=True)))
                
     
            pred_disp = pred_disp_0.cpu()[:, 0].numpy()
            #pred_disp_viz = pred_disp_0.squeeze()

            if opt.post_process:
                N = pred_disp.shape[0] // 2
                pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

            if opt.eval_sky:
                sky_mask = toNumpy(data['inputMask'].to(device).cpu(), keepDim=True)
                sky_masks.append(sky_mask)

            # gt_rsz = fn.interpolate(gt, size=(480, 640), mode='nearest')
            # corrLoss = 1 - corr_loss(input_color, gt_rsz, depth)

            pred_disps.append(pred_disp)
            input_colors.append(toNumpy(input_color.cpu(), keepDim=True))
            gt_depths.append((gt.cpu()))
            
            if saveFig:
                BG_R = torch.max(input_color[:,1:, :,:], dim=1, keepdim=True)[0] - torch.unsqueeze(input_color[:,0,:,:], dim=1)
                b = toNumpy(BG_R).flatten()
                d = toNumpy(depth).flatten()
                gt_rsz = fn.interpolate(gt, size=(480, 640), mode='nearest')
                g = toNumpy(gt_rsz).flatten() # gc = gt; dc = pred; bc = bgr
                gc = g[g>0]; bc = b[g>0]/4; dc = d[g>0]*4
                ds = 5000
                from scipy import stats
                corr_gt2pred = stats.pearsonr(gc, dc)
                corr_gt2ulap = stats.pearsonr(gc, bc)
                corr_pred2ulap = stats.pearsonr(dc, bc)
                tot_corr_gt2pred+=corr_gt2pred[0]
                tot_corr_gt2ulap+=corr_gt2ulap[0]
                tot_corr_pred2ulap+=corr_pred2ulap[0]
                for ptt in range(1, gc.shape[0], ds):
                    if dc[ptt]<8 and gc[ptt]<8:
                        #plt.plot(dc[ptt], bc[ptt], '.',color='r')
                        plt.plot(gc[ptt], bc[ptt], '.',color='b')
                        # plt.pause(0.05)

        if saveFig:
            #plt.plot(dc[ptt], bc[ptt], '.',color='r', label="predicted depth")
            plt.plot(gc[ptt], bc[ptt], '.',color='b', label="ground truth points")
            plt.legend(loc="lower right")
            plt.xlabel("Depth Points [m]")
            plt.ylabel("ULAP")
            # plt.show()
            fig.savefig(save_dir + '/ulap2GTDepthPlot.png')

        # end_time = time.time()
        # inferring = end_time - init_time
        # print("===>total time:{}".format(sec_to_hm_str(inferring)))

    pred_disps = np.concatenate(pred_disps)
    input_colors = np.concatenate(input_colors)
    gt_depths = np.concatenate(gt_depths)
    if opt.use_recons_net:
        pred_Js = np.concatenate(pred_Js)
    if opt.eval_sky:
        sky_masks = np.concatenate(sky_masks)
 
    tot_corr_gt2pred/=i
    tot_corr_gt2ulap/=i
    tot_corr_pred2ulap/=i

    # save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions"+opt.model_name)
    # print("-> Saving out benchmark predictions to {}".format(save_dir))
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

 
    output_path = os.path.join(
            opt.load_weights_folder, "benchmark_predictions"+opt.model_name, "disps_{}_split.npy".format(opt.eval_split))
    print("-> Saving predicted disparities to ", output_path)
    np.save(output_path, pred_disps)




    print("-> Evaluating")

    if opt.eval_sky:
        sky_errs=0
        skyErrList = []
    errors = []
    ratios = []
    plt.set_cmap('jet')  
    tobe_cleaned = []
    cleaned = list(range(pred_disps.shape[0]))
    for i in tobe_cleaned:
        if i in cleaned:
            cleaned.remove(i)
    sky_count=0
    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_depth = np.squeeze(gt_depth)
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        depth4J = 5 / (pred_disp+1e-3)
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp
 
        mask = gt_depth > 0
        
        # outDepth = cv2.resize(pred_depth, (opt.width, opt.height))
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
        _outDepth = 1 / pred_disps[i]
        outDepth = (normalize_numpy(_outDepth)*255).astype(np.uint8)
        inGT = (normalize_numpy(gt_depths[i])*255).astype(np.uint8)
        inGT = np.squeeze(inGT)
        inputColor = input_colors[i]
        inDisp = gt_depths[i].copy()
        inDisp[inDisp>0] = 1 / inDisp[inDisp>0]
        inDisp = np.squeeze(normalize_numpy(inDisp)*255).astype(np.uint8)
        # depth = 32.779243 / disp_resized
        # depth = np.clip(depth, 0, 80)/10
        # depth = np.uint8(depth * 256)
        save_path = os.path.join(save_dir, "{:010d}.png".format(i))

        if opt.use_recons_net:
            pred_Js
            J = pred_Js[i]
            plt.imsave(save_dir + "/frame_{:06d}_color_recons.jpg".format(i), J)

        plt.imsave(save_dir + "/frame_{:06d}_color.jpg".format(i), inputColor)
        plt.imsave(save_dir + "/frame_{:06d}_disp.png".format(i), outPred)
        plt.imsave(save_dir + "/frame_{:06d}_gt.png".format(i), inGT)
        plt.imsave(save_dir + "/frame_{:06d}_inDisp.png".format(i), inDisp)
        plt.imsave(save_dir + "/frame_{:06d}_outDepth.png".format(i), outDepth)

        color = (inputColor*255).astype(np.uint8)

        if opt.eval_sky:
            inGTMask = cv2.resize(inGT, (outPred.shape[1], outPred.shape[0]))
            inGTMask[inGTMask>0]=255
            inGTMask = inGTMask.astype(np.uint8)
            kernel = np.ones((5, 5), 'uint8')       
            inGTMask = cv2.dilate(inGTMask, kernel, iterations=1)
            inGTMask = 255- inGTMask
            sky_mask = np.squeeze(sky_masks[i])
            sky_mask = (np.sum(sky_mask,2)/3*255).astype(np.uint8)
            sky_mask[inGTMask<255]=0
            if np.any(sky_mask):
                sky_count+=1
                maxDepth = np.max(gt_depths[i])
                height, width = sky_mask.shape[:2]
                # pred_depth = 1 / pred_disp
                # pred_depth *= ratio
                # pred_depth = cv2.resize(pred_depth, (width, height))
                # sky_abs_err = 1/(pred_depth[sky_mask>0])
                
                pred_disp_sc = cv2.resize(pred_disp, (width, height))
                sky_abs_err = pred_disp_sc[sky_mask>0]

                sky_abs_err = sky_abs_err[sky_abs_err>=0]
                sky_err = np.mean(sky_abs_err)
                # print(sky_err)
                plt.imsave(save_dir + "/frame_{:06d}_sky_mask.bmp".format(i), sky_mask)
                sky_errs+=sky_err
                skyErrList.append(sky_err)


        cmap = plt.cm.jet

        def depth_colorize(depth):
            depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
            depth = 255 * cmap(depth)[:, :, :3]  # H, W, C
            return depth.astype('uint8')

        img_list = []
        img_list.append(color)
        img_list.append(depth_colorize(cv2.resize(inGT, (outPred.shape[1], outPred.shape[0]))))
        img_list.append(depth_colorize(outDepth))
        img_merge = np.hstack(img_list)
        plt.imsave(save_dir + "/frame_{:06d}_res_comparison.bmp".format(i), img_merge)
        ##

                ## debug A
        if 0:
            img = inputColor
            A = estimateA(img, depth4J)
            TM = np.zeros_like(img)
            for t in range(3):
                # TM[:,:,t] =  np.exp(-beta_rgb[t]*depth)
                TM[:,:,t] =  water_types_Nrer_rgb["3C"][t]**depth4J
            S = A*(1-TM)
            J = normalize_numpy((img - A) / TM + A)
            Sn = (normalize_numpy(S)*255).astype(np.uint8)
            plt.imsave(save_dir + "/frame_{:06d}_J.bmp".format(i), J)
            plt.imsave(save_dir + "/frame_{:06d}_S.bmp".format(i), Sn)


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
        "tot_corr_gt2pred": np.round(tot_corr_gt2pred,3),
        "tot_corr_gt2ulap": np.round(tot_corr_gt2ulap,3),
        "tot_corr_pred2ulap": np.round(tot_corr_pred2ulap,3)
        }

    if opt.eval_sky:
        sky_errs/=sky_count
        resdict = {
            "time": time,
            "abs_rel": np.round(mean_errors[0],3),
            "sq_rel": np.round(mean_errors[1],3),
            "rmse": np.round(mean_errors[2],3),
            "rmse_log": np.round(mean_errors[3],3),
            "a1": np.round(mean_errors[4],3),
            "a2": np.round(mean_errors[5],3),
            "a3": np.round(mean_errors[6],3),
            "sky_err": np.round(sky_errs,3),
            "tot_corr_gt2pred": np.round(tot_corr_gt2pred,3),
            "tot_corr_gt2ulap": np.round(tot_corr_gt2ulap,3),
            "tot_corr_pred2ulap": np.round(tot_corr_pred2ulap,3)
            }
        print(f"skyError: {sky_errs}")
        # with open("skyErrs_kitti.txt", 'w') as f:
        #     for s in skyErrList:
        #         f.write(str(s) + '\n')

    writer.writerow(resdict.keys())
    writer.writerow(resdict.values())

    f.close()

if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
