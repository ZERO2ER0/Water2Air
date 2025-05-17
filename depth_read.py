import os
import glob

import numpy as np

from sea_thru_utils import *


# def NYUv2_depth_read(image_path, preprocess=False):
#     dataset_dir = os.path.dirname(os.path.dirname(image_path))
#     depth_name = os.path.basename(image_path).replace('image', 'depth') 
        
#     depth_path = os.path.join(dataset_dir, 'depths', depth_name)
#     depth = np.load(depth_path)
#     # pdb.set_trace()
#     return depth
    # related_dir = os.path.abspath(related_dir)
    # pdb.set_trace()

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
    depth = depth - np.min(depth) + 100 # 平移整个深度图
    
    if preprocess == True:
        print(f"{depth_path}正在处理")
        # pdb.set_trace()
        depth = cv2.resize(depth, dsize = (image.shape[1], image.shape[0]), interpolation = cv2.INTER_CUBIC)
        # print(np.min(depth), np.max(depth), np.isnan(depth).sum(), np.isinf(depth).sum(), np.any(depth<=0)) 
        depth = np.max(depth) / depth # disparity map to depth map
        # depth = np.nan_to_num(depth, nan=np.finfo(np.float32).eps, posinf=np.max(depth), neginf=np.min(depth))
        depth = preprocess_predicted_depths(image, depth)
        # if not np.isfinite(depth).all():
        #     pdb.set_trace()
        

    # depth = normalize_depth_map(depth, 0.1, 10.0)
    return depth