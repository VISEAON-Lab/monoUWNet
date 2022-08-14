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
# dname='flatiron'
# date='20220628'
dname='FLC_4DS_tiny_sky'
date='20220706'
# dtype = 'flatiron'
run_name=date + '_FLC_'

runs = ['', 'corrLoss_', 'lvw_', 'rhf_', 'all_', 'all_wo_corrLoss_', 'all_wo_lvw_', 'all_wo_rhf_']
# runs = ['', 'corrLoss_', 'lvw_', 'rhf_', 'all_', 'all_wo_corrLoss_', 'all_wo_lvw_']
fullnameRuns = [run_name + i+dname for i in runs]
run1Name = fullnameRuns[0]
data_sets = ['_uc']#, '_flatiron', '_tiny']
for data in data_sets:
# data='_uc'
# data=''
    folder1 = '/home/samitai/Work/myDIFFNet/models/{}/models/weights_last/benchmark_predictions{}_eval{}'.format(run1Name, run1Name, data)
    outFolder = os.path.join('/home/samitai/Work/myDIFFNet/Comparisons', folder1.split('/')[-1] + '_VS_' + 'others')
    if not os.path.exists(outFolder):
            os.makedirs(outFolder)
    
    resultFiles1 = glob.glob(os.path.join(folder1, '*disp.{}'.format('png')))

    for file1 in resultFiles1:
        try:
            imageName = file1.split('/')[-1]
            colorName = imageName.replace('disp.png', 'color.jpg')
            gtName = imageName.replace('disp', 'inDisp')
            colorFile1 = os.path.join(folder1, colorName)
            gtFile1 = os.path.join(folder1, gtName)
            
            # if not os.path.isfile(file2):
            #     continue
            color1 = pil_loader(colorFile1)
            gt1 = pil_loader(gtFile1)
            gt1 = gt1.resize((color1.width, color1.height))
            disp1 = pil_loader(file1)
            
            img_list = []
            img_list.append(color1)
            img_list.append(gt1)
            img_list.append(disp1)
            for runiName in fullnameRuns[1:]:
                folder_i = '/home/samitai/Work/myDIFFNet/models/{}/models/weights_last/benchmark_predictions{}_eval{}'.format(runiName, runiName, data)
                file_i = os.path.join(folder_i, imageName)
                disp_i = pil_loader(file_i)

                img_list.append(disp_i)

            img_merge = np.vstack(img_list)
            plt.imsave(outFolder + "/{}_res_comparison.png".format(imageName), img_merge)

        except:
            print('file not found in ' + file1)