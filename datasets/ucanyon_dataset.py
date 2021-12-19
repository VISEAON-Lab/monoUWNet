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

from .mono_dataset import MonoDataset

class UCanyonDataset(MonoDataset):
    """Superclass for different types of south carolina dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(UCanyonDataset, self).__init__(*args, **kwargs)

        self.full_res_shape = (968, 608)
        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        self.K = np.array([[1215.4715960880724/self.full_res_shape[0], 0, 423.99891909157924/self.full_res_shape[0], 0],
                           [0, 1211.2257944573676/self.full_res_shape[1], 293.91172138607783/self.full_res_shape[1], 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        # line = self.filenames[0].split(',')
        # frameIndex = int(line[0])
        # frame_name = (line[1])

        # img_filename = os.path.join(
        #     self.data_path,frame_name)

        # return os.path.isfile(img_filename)
        return False

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

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
        f_str = f_str[:-5]+'_abs_depth.tif'
        image_path = os.path.join(
            self.data_path, 'depth',
            f_str)
        return image_path

    
    def get_depth(self, folder, frame_index, side, do_flip):
        depth_path = self.get_depth_path(folder, frame_index, side)
        try:
            depth_gt = pil.open(depth_path)
        except:
            return None
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32)
        depth_gt = preProcessDepth(depth_gt)

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt 
        

