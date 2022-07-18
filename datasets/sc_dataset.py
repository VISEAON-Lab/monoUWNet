# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import cv2
from my_utils import *
import glob
from .mono_dataset import MonoDataset
from utils import homorphicFiltering

class SCDataset(MonoDataset):
    """Superclass for different types of south carolina dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(SCDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        self.K = np.array([[1138.6334240599165/1600, 0, 720.1812627366363/1600, 0],
                           [0, 1138.774613252216/1200, 602.906549415305/1200, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1600, 1200)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        # line = self.filenames[0].split(',')
        # frameIndex = int(line[0])
        # frame_name = (line[1])

        # img_filename = os.path.join(
        #     self.data_path,frame_name)

        # return os.path.isfile(img_filename)
        return False

    def get_color(self, folder, frame_index, side, do_flip, x_hf=None):
        color_path = self.get_image_path(folder, frame_index, side)
        color = self.loader(color_path)
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        
        if self.use_hf and x_hf!=-1:
            # hf_path = os.path.join(self.data_path, 'imgs_hf')
            # if not os.path.exists(hf_path):
            #     os.makedirs(hf_path)
            # color_hf_path = color_path.replace("imgs", "imgs_hf")
            # if not os.path.isfile(color_hf_path):
            G = None
            #     if os.path.isfile(os.path.join(self.data_path, "G.npy")):
            #         G = np.load(os.path.join(self.data_path, "G.npy"))
            hf_color = pil.fromarray(homorphicFiltering(color, G, x_hf))
            #     # hf_color.save(color_hf_path)
            # else:
            #     hf_color = self.loader(color_hf_path)
            return hf_color

        return color

    def get_image_path(self, folder, frame_index, side):
        idx, frameName = folder.split(',')
        idx = int(idx)
        try:
            frameName = self.filenames[idx+frame_index].split(',')[1]
        except:
            frameName = self.filenames[idx].split(',')[1]
        f_str = frameName
        image_path = os.path.join(
            self.data_path, 'imgs',
            f_str)
        return image_path

    def get_depth_path(self, folder, frame_index, side):
        idx, frameName = folder.split(',')
        idx = int(idx)
        try:
            frameName = self.filenames[idx+frame_index].split(',')[1]
        except:
            frameName = self.filenames[idx].split(',')[1]
        f_str = frameName
        f_str = f_str[3:-8]+'*_SeaErra_abs_depth.tif'
        image_path = os.path.join(
            self.data_path, 'depth',
            f_str)
        files = glob.glob(image_path)
        # print(image_path)
        return files[0]

    
    def get_depth(self, folder, frame_index, side, do_flip):
        depth_path = self.get_depth_path(folder, frame_index, side)
        try:
            depth_gt = pil.open(depth_path)
        except:
            return None
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32)
        # depth_gt = preProcessDepth(depth_gt)

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt 
        

