from genericpath import exists
import numpy as np
import cv2
from my_utils import *
import glob
import os
from PIL import Image  # using pillow

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

folder1 = '/Users/shlomiamitai/work/myRepo/ANSFL/monocularDepthNN/myDIFFNet/models/20220122_sc_train/models/weights_last/benchmark_predictions20220122_sc_train evaluation'
folder2 = '/Users/shlomiamitai/work/myRepo/ANSFL/monocularDepthNN/myDIFFNet/models/20220315_sc_train_weighted_reproj_loss/models/weights_last/benchmark_predictions20220315_sc_train_weighted_reproj_loss'
outFolder = os.path.join('/Users/shlomiamitai/work/myRepo/ANSFL/monocularDepthNN/myDIFFNet/Comparisons', folder1.split('/')[-1] + '_VS_' + folder2.split('/')[-1])
if not os.path.exists(outFolder):
        os.makedirs(outFolder)

resultFiles1 = glob.glob(os.path.join(folder1, '*disp.{}'.format('bmp')))

for file1 in resultFiles1:
    imageName = file1.split('/')[-1]
    colorName = imageName.replace('disp', 'color')
    colorFile1 = os.path.join(folder1, colorName)
    file2 = os.path.join(folder2, imageName)
    if not os.path.isfile(file2):
        continue
    color1 = pil_loader(colorFile1)
    disp1 = pil_loader(file1)
    disp2 = pil_loader(file2)
    img_list = []
    img_list.append(color1)
    img_list.append(disp1)
    img_list.append(disp2)
    img_merge = np.hstack(img_list)
    plt.imsave(outFolder + "/{}_res_comparison.bmp".format(imageName), img_merge)

