import os
import math
import pickle
import random
import numpy as np
import torch
import cv2
import copy
from itertools import combinations

####################
# Files & IO
####################

###################### get PIPAL image combinations ######################
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def all_img_paths(img_root):
    assert os.path.isdir(img_root), '{:s} is not a valid directory'.format(img_root)
    images_paths = {}
    for dirRoot, _, fnames in os.walk(img_root):
        for fname in fnames:
            if is_image_file(fname):
                images_paths[fname] = os.path.join(dirRoot, fname)
    return images_paths


def image_combinations(ref_root, dist_root, mos_root, phase='train', dataset_name='PIPAL'):
    if phase == 'train':
        img_extension = '.bmp'
        ''' Form Combinations :  [ [ref_path, dist_A_path, dist_B_path, real_pro ], [..], ....] '''
        assert os.path.isdir(ref_root) is True, '{} is not a valid directory'.format(ref_root)
        assert os.path.isdir(dist_root) is True, '{} is not a valid directory'.format(dist_root)
        assert os.path.isdir(mos_root) is True, '{} is not a valid directory'.format(mos_root)

        '''obtain name of ref'''
        ref_fnames = list(map(lambda x: x if is_image_file(x) else print('Ignore {}'.format(x)), sorted(list(map(lambda x: x, os.walk(ref_root)))[0][2])))

        '''obtain paths of distortions. dict contains 200 list, every list contains different distorted image of one reference '''
        dist_class_Ref = {ref_name: [] for ref_name in ref_fnames}
        for dirpath, _, fnames in os.walk(dist_root):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    ref_name = fname.split("_")[0]+img_extension
                    dist_class_Ref[ref_name].append(fname)

        '''obtain MOS score of every distortion img'''
        mos_dict = {}
        for root, _, fnames in os.walk(mos_root):
            for fname in fnames:
                if ".txt" in fname:
                    mos_path = os.path.join(mos_root,fname)
                    with open(mos_path, "r") as f_ELO:
                        lines = f_ELO.readlines()
                        splited_lines = [dist_score.split(',') for dist_score in lines]
                        for DisName_ELO in splited_lines:
                            DisName, dist_ELO = DisName_ELO[0], float(DisName_ELO[1][:-1])
                            mos_dict[DisName] = dist_ELO

        ''' obtain [dist_A, dist_B, ref, real_pro] '''
        pair_combinations = {ref_name: [] for ref_name in ref_fnames}
        for ref_fname, dist_paths in dist_class_Ref.items():
            distAB_combinations = [list(dist_AB) for dist_AB in combinations(dist_paths, 2)]
            for index, dist_AB in enumerate(distAB_combinations):
                # Obtain [ dist_A, dist_B, ref ]
                dist_AB.append(dist_AB[0].split("_")[0] + img_extension)
                # Obtain [ dist_A, dist_B, ref, real_pro ]
                if dist_AB[0] in mos_dict and dist_AB[1] in mos_dict:
                    dist_A_score = mos_dict[dist_AB[0]]
                    dist_B_score = mos_dict[dist_AB[1]]
                    dist_AB.append(dist_A_score)
                    dist_AB.append(dist_B_score)
                else:
                    print(index, dist_AB)
                    raise NotImplementedError("There is Distorted Image that does not have MOS scores in Your MOS file!")
            pair_combinations[ref_fname] = distAB_combinations

        names_ref, names_dist_A, names_dist_B, dist_A_scores, dist_B_scores = [], [], [], [], []
        for _, pairs in pair_combinations.items():
            for pair in pairs:
                names_dist_A.append(pair[0])
                names_dist_B.append(pair[1])
                names_ref.append(pair[2])
                dist_A_scores.append(pair[3])
                dist_B_scores.append(pair[4])

        return names_ref, names_dist_A, names_dist_B, dist_A_scores, dist_B_scores


    elif phase == 'test':
        '''Prepare Testing or Validation Image Name List'''
        if dataset_name != 'PIPAL' and "TID2013":
            raise NotImplementedError('Nor PIPAL or TID2013. Please check dataset name in configuration file.')
        ref_fnames, dist_fnames = [], []
        for dirpath, _, fnames in os.walk(dist_root):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    img_extension = fname.split(".")[-1]
                    ref_fnames.append(fname.split("_")[0] + '.' + img_extension)
                    dist_fnames.append(fname)
        return ref_fnames, dist_fnames
    else:
        raise NotImplementedError("Error: Wrong Phase ! Onlhy Train and Test")




###################### get BAPPS 2AFC image combinations ######################
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

NP_EXTENSIONS = ['.npy', ]


def is_image_file(filename, mode='img'):
    if (mode == 'img'):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
    elif (mode == 'np'):
        return any(filename.endswith(extension) for extension in NP_EXTENSIONS)


def make_dataset(dirs, mode='img'):
    if (not isinstance(dirs, list)):
        dirs = [dirs, ]

    images = []
    for dir in dirs:
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname, mode=mode):
                    path = os.path.join(root, fname)
                    images.append(path)

    # print("Found %i images in %s"%(len(images),root))
    return images

###################### read images ######################
def read_img(path, size=None):
    '''
    read image by cv2 or from lmdb
    return: Numpy float32, HWC, BGR, [0,1]
    '''
    # resize to 256 x 256


    # read image
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if size and img.shape[0] > size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)

    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


###### img augument  ##############
def translate_img(img, max_shift=3.5):
    height, width = img.shape[:2]
    x_shift = random.uniform(-max_shift, max_shift)
    y_shift = random.uniform(-max_shift, max_shift)
    matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    trans_img = cv2.warpAffine(img, matrix, (width, height))
    return trans_img


####################
# image processing
# process on numpy image
####################

def channel_convert(in_c, tar_type, img_list):
    # conversion among BGR, gray and y
    if in_c == 3 and tar_type == 'gray':  # BGR to gray
        gray_list = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in img_list]
        return [np.expand_dims(img, axis=2) for img in gray_list]
    elif in_c == 3 and tar_type == 'y':  # BGR to y
        y_list = [bgr2ycbcr(img, only_y=True) for img in img_list]
        return [np.expand_dims(img, axis=2) for img in y_list]
    elif in_c == 1 and tar_type == 'RGB':  # gray/y to BGR
        return [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in img_list]
    else:
        return img_list


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def bgr2ycbcr(img, only_y=True):
    '''bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)



if __name__ == '__main__':
    ref_root = '/home/jjgu/home/hmcai/IQA_Results/PIPAL/Origins/PIPAL_Origin'
    dist_root = '/home/jjgu/home/hmcai/IQA_Results/PIPAL/Distortions'
    mos_root = '/home/jjgu/home/hmcai/IQA_Results/PIPAL/EloMOSV1.txt'
    phase = 'Train'
    image_combinations_pipal(ref_root, dist_root, mos_root, phase)

