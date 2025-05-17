import os, sys, pdb
import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch.nn.functional as F

from PIL import Image

from sea_thru_utils import read_folder_file
from sea_thru import SAUD_depth_read

from NYUv2_prep import write_depth

def U45_depth_statistics(dataset_folder):
    images_folder = os.path.join(dataset_folder, 'images')
    images_path = read_folder_file(images_folder)
    
    depths = []
    depths_max = []
    depths_min = []    
    for image_path in images_path:
        depth = U45_depth_read(image_path, preprocess=False)
        # pdb.set_trace()
        
        depths.append(depth)
        depths_max.append(np.max(depth))
        depths_min.append(np.min(depth))
        
    
    pdb.set_trace()# print(f"Depth statistics: Max={max(depths)}, Min={min(depths)}, Mean={sum(depths) / len(depths)}")


def U45_depth_pre(dataset_folder, child_folder, model):
    images_folder = os.path.join(dataset_folder, child_folder)
    images_path = read_folder_file(images_folder)
    
    depth_folder = os.path.join(dataset_folder, 'depths_zoe')
        
    depths = []
    depths_max = []
    depths_min = []    
    for image_path in images_path:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        depth_path = os.path.join(depth_folder, image_name + '.npy')
        depth_vis_path = os.path.join(depth_folder, image_name)
        
        image = Image.open(image_path)
        output = model(image)
        pre_depth = output['predicted_depth'].numpy()
        # pdb.set_trace()
        np.save(depth_path, pre_depth)
        write_depth(depth_vis_path, pre_depth, grayscale = False)
        # output['depth'].save(depth_vis_path)
        
        # pre_depth = zeodepth_postprocessing(pre_depth, image)
        depths.append(pre_depth)
        depths_max.append(np.max(pre_depth))
        depths_min.append(np.min(pre_depth))
        
    
    # pdb.set_trace()# 

def zoedepth():
    from transformers import pipeline

    pipe = pipeline(task="depth-estimation", 
                    model="/home/chenli/model/transformers/zoedepth-nyu-kitti", 
                    device='cuda:1')
    
    return pipe


# def zeodepth_postprocessing(predicted_depth, image):
#     prediction = F.interpolate(
#         predicted_depth.unsqueeze(1),
#         size=image.size[::-1],
#         mode="bicubic",
#         align_corners=False,
#     )
#     return prediction


    # result = pipe(image)
    # depth = result["depth"]
    # # depth = Image.fromarray(depth.astype("uint8"))
    # depth.save('depth_test.png')

if __name__ == '__main__':
    # dataset_folder = '/home/chenli/data/UW/SAUD/Benchmark/Best_bt'
    # SAUD_depth_statistics(dataset_folder)
        
        
    dataset_folder = '/home/chenli/data/UW/U45'
    U45_depth_pre(dataset_folder, 'images', zoedepth())
    