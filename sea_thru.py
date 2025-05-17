# 
# This file is part of Sea-Thru-Impl.
# Copyright (c) 2022 Zeyuan HE (Teragion).
# 
# This program is free software: you can redistribute it and/or modify  
# it under the terms of the GNU General Public License as published by  
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License 
# along with this program. If not, see <http://www.gnu.org/licenses/>.
# 

import argparse
import ctypes
import json, pickle
import sys, os, glob, pdb

import numpy as np
import math
import queue

import sklearn as sk
import scipy
import scipy.optimize
import scipy.stats

import cv2
from PIL import Image
import rawpy

from sea_thru_utils import * 

def calculate_Ja(raw_file_path, mode='Map', hint = None, size = None, prefix = None):
    prefix = (prefix if (prefix is not None) else "")
    Ba, coeffs_B, coeffs_D, att, Ea, Da, depths = load_template_params(raw_file_path, mode, hint, size, prefix)
    # pdb.set_trace()
    E = np.uint8(np.clip(Ea, 0, 1) * 255.0)
    illuminant_map = Image.fromarray(E)
    illuminant_map.save("out/" + prefix + "illuminant_map.png")

    Ja = recover(Da, att, depths)

    Ja = Ja / np.max(Ja)
    Ja = exposure.equalize_adapthist(Ja)

    Ja *= 255.0
    # Ja = np.uint8(Ja)
    Js = wb_ycbcr_mean(Ja)

    result = Image.fromarray(Js)
    result.save("out/" + prefix + "out.png")
    print("Finished.")
    
    
def cal_Ja_folder(image_folder, depth_read_func, exclude_images):
    images_path = read_folder_file(image_folder)
    # depths_path = read_folder_file(depth_folder)

    for image_path in sorted(images_path):
        print(f"{image_path} params calculate start.")
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        if image_name in exclude_images:
            continue
        image = read_image(image_path)
        depth = depth_read_func(image_path, False)
        
        params_path = rawfile_path2related_path(image_path, 'params_zoe_norm')
        if os.path.exists(params_path):
            with open(params_path, 'rb') as f:
                params = pickle.load(f)
                Ba = params['Ba']
                coeffs_B = params['coeffs_B']
                coeffs_D = params['coeffs_D']
                att = params['att']
                Ea = params['Ea']
            Da = image - Ba
            Da = np.clip(Da, 0, 1)
            
            D = np.uint8(Da * 255.0)
            backscatter_removed = Image.fromarray(D)
            # backscatter_removed.save("out/" + prefix + "direct_signal.png")
        else:
            # print("Estimating backscatter...")
            # pdb.set_trace()
            Ba, coeffs_B = estimate_backscatter(image, depth)

            Da = image - Ba
            Da = np.clip(Da, 0, 1)

            D = np.uint8(Da * 255.0)
            backscatter_removed = Image.fromarray(D)
            # backscatter_removed.save("out/" + prefix + "direct_signal.png")

            # print("Estimating wideband attenuation...")
            att, Ea, coeffs_D = estimate_wideband_attenuation(Da, depth)
            params = {'Ba': Ba,
                    'coeffs_B': coeffs_B,
                    'coeffs_D': coeffs_D,
                    'att': att,
                    'Ea': Ea}
            save_params(params, image_path, 'params_zoe_norm')
            
            
        E = np.uint8(np.clip(Ea, 0, 1) * 255.0)
        illuminant_map = Image.fromarray(E)
        # illuminant_map.save("out/" + "illuminant_map.png")

        Ja = recover(Da, att, depth)

        Ja = Ja / np.max(Ja)
        Ja = exposure.equalize_adapthist(Ja)

        Ja *= 255.0
        # Ja = np.uint8(Ja)
        Js = wb_ycbcr_mean(Ja)

        result = Image.fromarray(Js)
        Ja_path = rawfile_path2related_path(image_path, 'sea_thru_zoe_norm')
        result.save(Ja_path)
        # print(f"{image_path} params calculate finished.")
        
def SAUD_depth_read(image_path, preprocess=False):
    # 替换路径中的 'images' 为 'depths'
    depth_path = image_path.replace('images', 'depths')

    # 使用 os.path.splitext 去掉扩展名部分
    depth_path = os.path.splitext(depth_path)[0]

    # 添加通配符 .pfm，构造完整路径
    depth_path_with_wildcard = depth_path + '-*.pfm'

    # 使用 glob 来查找符合条件的 .pfm 文件
    matching_depth_files = glob.glob(depth_path_with_wildcard)

    if len(matching_depth_files) == 0:
        raise FileNotFoundError(f"No matching .pfm files found for {depth_path_with_wildcard}")
    elif len(matching_depth_files) == 1:
        depth_path = matching_depth_files[0]
    else:
        print(matching_depth_files)
        raise FileExistsError(f"Multiple {depth_path_with_wildcard} exists.")
    
    image = read_image(image_path)
    # pdb.set_trace()
    depth, scale = read_pfm(depth_path)
    # depth = depth - np.min(depth) + 100 # 平移整个深度图
    
    if preprocess == True:
        print(f"{depth_path}正在处理")
        # pdb.set_trace()
        depth = cv2.resize(depth, dsize = (image.shape[1], image.shape[0]), interpolation = cv2.INTER_CUBIC)
        depth = depth - np.min(depth) + 100 # 平移整个深度图
        depth = depth.max() / depth
        # print(np.min(depth), np.max(depth), np.isnan(depth).sum(), np.isinf(depth).sum(), np.any(depth<=0)) 
        # depth = np.nan_to_num(depth, nan=np.finfo(np.float32).eps, posinf=np.max(depth), neginf=np.min(depth))
        depth = preprocess_predicted_depths(image, depth)
        # if not np.isfinite(depth).all():
        #     pdb.set_trace()
        

    # depth = normalize_depth_map(depth, 0.1, 10.0)
    return depth


def SAUD_zoe_depth(image_path, preprocess=False):
    # 替换路径中的 'images' 为 'depths'
    depth_path = image_path.replace('images', 'depths_zoe')

    # 使用 os.path.splitext 去掉扩展名部分
    depth_path = os.path.splitext(depth_path)[0]

    # 添加通配符 .pfm，构造完整路径
    depth_path_with_wildcard = depth_path + '.npy'

    # 使用 glob 来查找符合条件的 .pfm 文件
    matching_depth_files = glob.glob(depth_path_with_wildcard)

    if len(matching_depth_files) == 0:
        raise FileNotFoundError(f"No matching .npy files found for {depth_path_with_wildcard}")
    elif len(matching_depth_files) == 1:
        depth_path = matching_depth_files[0]
    else:
        print(matching_depth_files)
        raise FileExistsError(f"Multiple {depth_path_with_wildcard} exists.")
    
    image = read_image(image_path)
    # pdb.set_trace()
    depth = np.load(depth_path)
    # depth = depth - np.min(depth) + 100 # 平移整个深度图
    
    if preprocess == True:
        print(f"{depth_path}正在处理")
        # depth = depth.max() / depth
        depth = preprocess_predicted_depths(image, depth)
        
    return depth

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--original', required = True, help = "Path to original image")
    # # parser.add_argument('--depth', required = False, help = "Path to depth map")
    # parser.add_argument('--mode', required = True, help = "Mode = {{Map, Predict, Hybrid}}")
    # parser.add_argument('--hint', required = False, help = "Path to depth map as hint")
    # parser.add_argument('--size', required = False, type = int, help = "Maximum side of image to shrink")
    # parser.add_argument('--prefix', required = False, help = "Prefix for output files")

    # args = parser.parse_args()
    
    # '''
    # python sea_thru.py --original '/home/chenli/data/UW/sea_thru/D1/linearPNG/T_S02951.png'  --mode Map   
    # '''
    # calulate_Ja(args.original, mode='Map', hint = None, size = None, prefix = None)
    
    # image_path = '/home/chenli/data/UW/SAUD/images/1_1.png'
    # SAUD_depth_read(image_path)
    
    image_path = '/home/chenli/data/UW/SAUD/Benchmark/Best_bt/images'
    # exclude_images = ['1_5', '1_13',    # MiDaS, with wrong depth
    #                   '2_1', '2_2', '2_9', '2_14', '2_15',
    #                   '3_1', '3_4',
    #                   '4_2', '4_4', '4_10', '4_12', '4_20',
    #                   '5_16',]
    exclude_images = ['1_8', '1_15', '1_17',
                      '2_2', '2_12', '2_13',
                      '3_2', '3_6', 
                      '4_2', '4_4', '4_10', '4_14', '4_17',
                      '5_11']
    # exclude_images = ['2_10', '2_14', '1_13', '4_12', '4_20', '3_1', '2_9', '2_15', '3_4', '1_5', '5_16', '2_1', '4_10', '2_2', '4_4', '4_2', '3_7', '4_9']
    cal_Ja_folder(image_path, SAUD_zoe_depth, exclude_images)
