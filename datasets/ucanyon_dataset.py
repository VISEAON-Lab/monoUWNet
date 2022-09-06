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
from utils import homorphicFiltering
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
        self.K_prev = np.array([[1215.4715960880724/self.full_res_shape[0], 0, 423.99891909157924/self.full_res_shape[0], 0],
                           [0, 1211.2257944573676/self.full_res_shape[1], 293.91172138607783/self.full_res_shape[1], 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.K = np.array([[1184/self.full_res_shape[0], 0, 482.3459505/self.full_res_shape[0], 0],
                           [0, 1184/self.full_res_shape[1], 289.101975/self.full_res_shape[1], 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.distCoeffs = np.array([-0.133974, 0.15196, 0.0359522, 0.000293189, -0.00212398])
        self.cameraMatrix = self.K[:3, :3].copy()
        self.cameraMatrix[0,:]*=self.full_res_shape[0]
        self.cameraMatrix[1,:]*=self.full_res_shape[1]
        self.newCameraMatrix, self.roi = cv2.getOptimalNewCameraMatrix(self.cameraMatrix, self.distCoeffs, self.full_res_shape, 1, self.full_res_shape)
        self.K[0,0] = self.newCameraMatrix[0,0]/(self.roi[2]-self.roi[0])
        self.K[0,2] = self.newCameraMatrix[0,2]/(self.roi[2]-self.roi[0])
        self.K[1,1] = self.newCameraMatrix[1,1]/(self.roi[3]-self.roi[1])
        self.K[1,2] = self.newCameraMatrix[1,2]/(self.roi[3]-self.roi[1])
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
        K = self.K[:3, :3].copy()
        K[0,:]*=self.full_res_shape[0]
        K[1,:]*=self.full_res_shape[1]
        # undistorted_img = cv2.undistort(np.array(color), self.cameraMatrix, self.distCoeffs, self.newCameraMatrix)
        # roi = self.roi
        # # undistorted_img = undistorted_img[roi[1]:roi[3], roi[0]:roi[2]]
        # self.full_res_shape = (undistorted_img.shape[1], undistorted_img.shape[0])
        # undistorted_img=pil.fromarray(undistorted_img)
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        if self.use_hf and x_hf!=-1:
            # hf_path = os.path.join(self.data_path, 'imgs_hf')
            # if not os.path.exists(hf_path):
            #     os.makedirs(hf_path)
            # color_hf_path = color_path.replace("imgs", "imgs_hf")
            # if not os.path.isfile(color_hf_path):
            G = None
            #     # if os.path.isfile(os.path.join(self.data_path, "G.npy")):
            #     #     G = np.load(os.path.join(self.data_path, "G.npy"))
            hf_color = pil.fromarray(homorphicFiltering(color, G, x_hf))
            #     hf_color.save(color_hf_path)
            # else:
            #     hf_color = self.loader(color_hf_path)
            
            if 0:
                color.save('rhf3_'+str(0)+'.png')
                for d0 in range(5, 200, 10):
                    hf_color = pil.fromarray(homorphicFiltering(color, G, d0))
                    hf_color.save('rhf3_'+str(d0)+'.png')

            return hf_color
        return color

    def get_image_path(self, folder, frame_index, side):
        idx, frameName = folder.split(',')
        idx = int(idx)
        if idx+frame_index<0:
            frame_index=0
        if idx+frame_index>len(self.filenames)-1:
            frame_index=0

        frameName = self.filenames[idx+frame_index].split(',')[1]

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
        depth_path = self.get_depth_path(folder, frame_index, side) # abs_depth
        try:
            seara_abs_depth_path = depth_path.replace("_abs_depth","_SeaErra_abs_depth")
            depth_gt = pil.open(seara_abs_depth_path)
        except:
            try:
                no_addition_depth_path = depth_path.replace("_abs_depth","")
                depth_gt = pil.open(no_addition_depth_path)
                        
            except:
                try:
                    depth_gt = pil.open(depth_path) # abs_depth
                except:
                    return None

        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32)
        # depth_gt = preProcessDepth(depth_gt)

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt 
        

    def get_mask(self, folder, frame_index, side, do_flip):
        idx, frameName = folder.split(',')
        idx = int(idx)
        try:
            frameName = self.filenames[idx+frame_index].split(',')[1]
        except:
            frameName = self.filenames[idx].split(',')[1]
        f_str = frameName
        f_str = f_str[:-5]+'_skyMask.png'
        mask_filename = os.path.join(
            self.data_path, 'imgs',
            f_str)
        try:
            mask = self.loader(mask_filename)
        except:
            mask = pil.new(mode = "RGB", size = (self.full_res_shape[0], self.full_res_shape[1]),
                           color = (0, 0, 0))
        if do_flip:
            mask = np.fliplr(mask)

        return mask

