from __future__ import absolute_import, division, print_function

import os
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import transformation_from_parameters
from utils import readlines
from options import MonodepthOptions
from datasets import KITTIOdomDataset
import datasets
import networks
import matplotlib.pyplot as plt

# from https://github.com/tinghuiz/SfMLearner
def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        xyzs.append(cam_to_world[:3, 3])
    return xyzs


# from https://github.com/tinghuiz/SfMLearner
def compute_ate(gtruth_xyz, pred_xyz_o):

    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz_o[0]
    pred_xyz = pred_xyz_o + offset[None, :]

    # Optimize the scaling factor
    scale = np.sum(gtruth_xyz * pred_xyz) / np.sum(pred_xyz ** 2)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = np.sqrt(np.sum(alignment_error ** 2)) / gtruth_xyz.shape[0]
    return rmse


def evaluate(opt):
    """Evaluate odometry on the KITTI dataset
    """
    device = torch.device("cpu" if opt.no_cuda else "cuda")
    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    # assert opt.eval_split == "odom_9" or opt.eval_split == "odom_10", \
    #     "eval_split should be either odom_9 or odom_10"

    # sequence_id = int(opt.eval_split.split("_")[1])

    # filenames = readlines(
    #     os.path.join(os.path.dirname(__file__), "splits", "odom",
    #                  "test_files_{:02d}.txt".format(sequence_id)))
    splits_dir = os.path.join(os.path.dirname(__file__), "splits")
    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "all_files.txt"))
    # dataset = KITTIOdomDataset(opt.data_path, filenames, opt.height, opt.width,
    #                          [0, 1], 4, is_train=False)

    if opt.dataset=='sc': # aqualoc
        dataset = datasets.SCDataset(opt.dataset, opt.data_path, filenames,
                                        opt.height, opt.width,
                                        [0, 1], 4, is_train=False, opts = opt) 
    elif opt.dataset=='uc': # aqualoc
        dataset = datasets.UCanyonDataset(opt.dataset, opt.data_path, filenames,
                                        opt.height, opt.width,
                                        [0, 1], 4, is_train=False, opts = opt)

    dataloader = DataLoader(dataset, opt.batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True, drop_last=False)

    pose_encoder_path = os.path.join(opt.load_weights_folder, "pose_encoder.pth")
    pose_decoder_path = os.path.join(opt.load_weights_folder, "pose.pth")

    pose_encoder = networks.ResnetEncoder(opt.num_layers, False, 2)
    pose_encoder.load_state_dict(torch.load(pose_encoder_path, map_location=torch.device(device)))

    pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc, 1, 2)
    pose_decoder.load_state_dict(torch.load(pose_decoder_path, map_location=torch.device(device)))

    pose_encoder.to(device)
    pose_encoder.eval()
    pose_decoder.to(device)
    pose_decoder.eval()

    pred_poses = []

    print("-> Computing pose predictions")

    opt.frame_ids = [0, 1]  # pose network only takes two frames as input

    with torch.no_grad():
        for inputs in dataloader:
            for key, ipt in inputs.items():
                inputs[key] = ipt.to(device)

            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in opt.frame_ids], 1)

            features = [pose_encoder(all_color_aug)]
            axisangle, translation = pose_decoder(features)

            pred_poses.append(
                transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy())

    pred_poses = np.concatenate(pred_poses)

    # gt_poses_path = os.path.join(opt.data_path, "poses", "{:02d}.txt".format(sequence_id))
    # gt_global_poses = np.loadtxt(gt_poses_path).reshape(-1, 3, 4)
    # gt_global_poses = np.concatenate(
    #     (gt_global_poses, np.zeros((gt_global_poses.shape[0], 1, 4))), 1)
    # gt_global_poses[:, 3, 3] = 1
    # gt_xyzs = gt_global_poses[:, :3, 3]

    # gt_local_poses = []
    # for i in range(1, len(gt_global_poses)):
    #     gt_local_poses.append(
    #         np.linalg.inv(np.dot(np.linalg.inv(gt_global_poses[i - 1]), gt_global_poses[i])))

    ates = []
    pts=[]
    num_frames = pred_poses.shape[0]
    track_length = 125
    local_xyzs = np.array(dump_xyz(pred_poses))
        # gt_local_xyzs = np.array(dump_xyz(gt_local_poses[i:i + track_length - 1]))
        # pts.append(local_xyzs)
        # ates.append(compute_ate(gt_local_xyzs, local_xyzs))

    save_path = os.path.join(opt.load_weights_folder, opt.eval_split+ "_poses.npy")
    np.save(save_path, pred_poses)
    print("-> Predictions saved to", save_path)

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(local_xyzs[:,0], local_xyzs[:,1], local_xyzs[:,2])
    plt.show()

    print("\n   Trajectory error: {:0.3f}, std: {:0.3f}\n".format(np.mean(ates), np.std(ates)))

  

if __name__ == "__main__":
    options = MonodepthOptions()
    if 0:
        evaluate(options.parse())
    else:
        pred_poses = np.load('FLC_poses.npy')
        local_xyzs = np.array(dump_xyz(pred_poses))
        dr = abs(np.diff(local_xyzs, axis=0))
        dr_mu = np.mean(dr, axis=0)
        print(dr_mu)
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(local_xyzs[:,0], local_xyzs[:,1], local_xyzs[:,2])
        plt.show()


# flc deltas: [0.0017622  0.00168929 0.00175339]
# sc deltas: [0.00149089 0.00203293 0.00153094]