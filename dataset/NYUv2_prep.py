import os, sys, pdb
import cv2

import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sea_thru_utils import read_folder_file


def NYUv2_imagepath_to_depthpath(image_path):
    dataset_dir = os.path.dirname(os.path.dirname(image_path))
    base_name = os.path.splitext(os.path.basename(image_path))[0].replace('image', 'depth')
    # depth_name = os.path.basename(image_path).replace('image', 'depth')
        
    depth_path = os.path.join(dataset_dir, 'depths', base_name+'.npy')
    return depth_path

def remove_white_border(image, extra_border = 1):
    # 找到图像非白色部分的边界
    # 假设白色像素是 (255, 255, 255)
    # 如果是其他颜色，也可以通过更改 threshold 来适应
    threshold = 255  # 白色的阈值，通常255是白色，这里设定稍微低一点来捕捉接近白色的像素
    # 创建掩码，标记所有像素是否接近白色
    white_mask = np.all(image >= threshold, axis=-1)
    
    # 找到完全是白色的行和列
    non_white_rows = np.any(~white_mask, axis=1)  # 找到有非白色像素的行
    non_white_cols = np.any(~white_mask, axis=0)  # 找到有非白色像素的列
    
    # 获取非白色行列的索引范围
    y_min, y_max = np.where(non_white_rows)[0][[0, -1]]
    x_min, x_max = np.where(non_white_cols)[0][[0, -1]]
    
    c_image = Image.fromarray(image).crop((x_min+extra_border, y_min+extra_border, x_max + 1 - extra_border, y_max + 1 - extra_border))
    
    return np.array(c_image), y_min+extra_border, y_max + 1 - extra_border, x_min+extra_border, x_max + 1 - extra_border

def write_depth(path, depth, grayscale, bits=1):
    """Write depth map to png file.

    Args:
        path (str): filepath without extension
        depth (array): depth
        grayscale (bool): use a grayscale colormap?
    """
    if not grayscale:
        bits = 1

    if not np.isfinite(depth).all():
        depth=np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        print("WARNING: Non-finite depth values present")

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.dtype)

    if not grayscale:
        out = cv2.applyColorMap(np.uint8(out), cv2.COLORMAP_INFERNO)

    if bits == 1:
        cv2.imwrite(path + ".png", out.astype("uint8"))
    elif bits == 2:
        cv2.imwrite(path + ".png", out.astype("uint16"))

    return

def crop_depth(depth, y_min, y_max, x_min, x_max):
    return depth[y_min: y_max, x_min: x_max]

def NYUv2_depth_statistics(dataset_dir = '/home/chenli/data/images_data/NYU_v2_label/no_border/depths'):
    depths_path = read_folder_file(dataset_dir)
    depths = []
    depths_max = []
    depths_min = []    
    for depth_path in depths_path:
        depth = np.load(depth_path)
        
        depths.append(depth)
        depths_max.append(np.max(depth))
        depths_min.append(np.min(depth))
    pdb.set_trace()
    
    # 数值上 depth[10, :] > depth[470, :], 可视化： 近处颜色深，


if __name__ == '__main__':
    # dataset_dir = '/home/chenli/data/images_data/NYU_v2_label/'
    # images_dir = os.path.join(dataset_dir, 'raw_folder', 'images')
    # c_image_dir = os.path.join(dataset_dir, 'no_border', 'images')
    # c_depth_dir = os.path.join(dataset_dir, 'no_border', 'depths')
    # c_vis_depth_dir = os.path.join(dataset_dir, 'no_border', 'depths_vis')
    
    # images_path = read_folder_file(images_dir)
    # for image_path in images_path:
    #     image_name = os.path.splitext(os.path.basename(image_path))[0]
    #     depth_path = NYUv2_imagepath_to_depthpath(image_path)
    #     c_image_path = os.path.join(c_image_dir, image_name + '.png')
    #     c_depth_path = os.path.join(c_depth_dir, image_name.replace('image', 'depth') + '.npy')
    #     c_vis_depth_path = os.path.join(c_vis_depth_dir, image_name.replace('image', 'depth'))
        
    #     image = np.load(image_path)
    #     depth = np.load(depth_path)
        
    #     c_image, y_min, y_max, x_min, x_max = remove_white_border(image)
    #     c_depth = crop_depth(depth, y_min, y_max, x_min, x_max)
        
    #     Image.fromarray(c_image).save(c_image_path)
    #     np.save(c_depth_path, c_depth)
    #     write_depth(c_vis_depth_path, c_depth, grayscale=False)
        # pdb.set_trace()
        # depth_path = NYUv2_imagepath_to_depthpath(image_path)
    # image_path = '/home/chenli/data/images_data/NYU_v2_label/images/image_0001.npy'
    # image = np.load(image_path)
    
    
    # real_image = remove_white_border(image)
    NYUv2_depth_statistics('/home/chenli/data/images_data/NYU_v2_label/raw_folder/depths')