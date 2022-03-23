from os import listdir
from os.path import isfile, join, split
import os
import random as rand
import glob
import shutil
import re
from sys import platform
from distutils.dir_util import copy_tree
import sys
from os import path
# from natsort import natsorted
import csv
  




dataPath=r'/Users/shlomiamitai/work/myRepo/ANSFL/monocularDepthNN/dataset/flatiron/imgs'
f = open(dataPath + '/all_files.txt', 'w', newline='')
testWriter = csv.writer(f)


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def getAllImages(dataPath):
    onlyImages = glob.glob(dataPath + '/*.tiff' )
    return onlyImages

def unique(list1): 
    # intilize a null list 
    unique_list = [] 
    # traverse for all elements 
    for x in list1: 
        # check if exists in unique_list or not 
        if x not in unique_list: 
            unique_list.append(x) 
    return unique_list



onlyImages = getAllImages(dataPath)

print(len(onlyImages))

testCount=0

for imFolder in sorted(onlyImages):
    ps = re.split('\\\|\/', imFolder)
    parentFolder = ps[-2]
    fileName = ps[-1]
    data = [testCount, fileName]
    testWriter.writerow(data)
    testCount+=1


print(testCount)


