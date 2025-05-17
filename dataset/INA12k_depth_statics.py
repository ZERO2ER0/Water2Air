import os, sys, pdb
import torch
from torch.utils.data import DataLoader, Dataset
import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch.nn.functional as F
from transformers import AutoImageProcessor, ZoeDepthForDepthEstimation

from PIL import Image

from sea_thru_utils import read_folder_file
from sea_thru import SAUD_depth_read

from dataset.NYUv2_prep import write_depth

def INA12k_depth_statistics(dataset_folder):
    images_folder = os.path.join(dataset_folder, 'images')
    images_path = read_folder_file(images_folder)
    
    depths = []
    depths_max = []
    depths_min = []    
    for image_path in images_path:
        depth = SAUD_depth_read(image_path, preprocess=False)
        # pdb.set_trace()
        
        depths.append(depth)
        depths_max.append(np.max(depth))
        depths_min.append(np.min(depth))
        
    
    pdb.set_trace()# print(f"Depth statistics: Max={max(depths)}, Min={min(depths)}, Mean={sum(depths) / len(depths)}")


def INA12k_depth_pre(images_folder, depths_folder, model):
    images_path = read_folder_file(images_folder)
    
        
    depths = []
    depths_max = []
    depths_min = []    
    for image_path in images_path:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        depth_path = os.path.join(depths_folder, image_name + '.npy')
        depth_vis_path = os.path.join(depths_folder, image_name)
        
        image = Image.open(image_path).convert('RGB')
        output = model(image)
        pre_depth = output['predicted_depth'].numpy()
        if pre_depth.shape[0] != np.array(image).shape[0]:
            print(image_name)
    
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
                    device='cuda:0')
    
    # model = ZoeDepthForDepthEstimation.from_pretrained("/home/chenli/model/transformers/zoedepth-nyu-kitti")
    
    return pipe

def INA12k_image_path2depth_path(image_path):
    # pdb.set_trace()
    dataset_dir = os.path.dirname(os.path.dirname(os.path.dirname(image_path)))
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    # depth_name = os.path.basename(image_path).replace('image', 'depth')
        
    depth_path = os.path.join(dataset_dir, 'depths', 'train', base_name+'.npy')
    return depth_path

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.endswith((".jpg", ".png", ".jpeg"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        # if self.transform:
        #     image = self.transform(image)
        return image_path, image
    
from torchvision import transforms
# # 定义自定义 collate_fn
# def custom_collate_fn(batch):
#     transform = transforms.ToTensor()
#     image_paths, images = zip(*batch)
#     images = [transform(img) for img in images]  # 将 PIL 图像转换为张量
#     return image_paths, torch.stack(images)
# 推理和后处理
def predict_depth(dataloader, model, processor, output_dir, device):
    model.to(device)
    model.eval()
    

    results = {}
    with torch.no_grad():
        for batch in dataloader:
            image_path, image = batch
            # pdb.set_trace()
            image_name = os.path.splitext(os.path.basename(image_path[0]))[0]
            depth_path = os.path.join(output_dir, image_name + '.npy')
            depth_vis_path = os.path.join(output_dir, image_name)
            
            image_pv = processor(image[0], return_tensors="pt").pixel_values
            image_pv = image_pv.to(device)

            # 模型推理
            outputs = model(pixel_values=image_pv)
            post_processed = processor.post_process_depth_estimation(
                outputs, 
                source_sizes= [(image.shape[1], image.shape[2])],
            )
            predicted_depth = post_processed[0]["predicted_depth"].cpu().numpy()
            
            np.save(depth_path, predicted_depth)
            write_depth(depth_vis_path, predicted_depth, grayscale = False)

            # 保存结果
            # for i, path in enumerate(image_paths):
            #     predicted_depth = post_processed[i]["predicted_depth"]
            #     depth = (
            #         (predicted_depth - predicted_depth.min())
            #         / (predicted_depth.max() - predicted_depth.min())
            #     )
            #     depth = depth.detach().cpu().numpy() * 255
            #     depth = Image.fromarray(depth.astype("uint8"))
            #     results[path] = depth

    return results

# 主函数
def main():
    # 设置路径
    image_dir = "/home/chenli/data/images_data/inaturalist_12K/images/val"  # 替换为你的数据集路径
    output_dir = "/home/chenli/data/images_data/inaturalist_12K/depths/val"  # 替换为保存深度图的路径
    os.makedirs(output_dir, exist_ok=True)

    # 初始化模型和处理器
    processor = AutoImageProcessor.from_pretrained("/home/chenli/model/transformers/zoedepth-nyu-kitti")
    model = ZoeDepthForDepthEstimation.from_pretrained("/home/chenli/model/transformers/zoedepth-nyu-kitti")

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据集
    dataset = ImageDataset(image_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # 推理
    results = predict_depth(dataloader, model, processor, output_dir, device)

    # 保存结果
    # for path, depth_image in results.items():
    #     save_path = os.path.join(output_dir, os.path.basename(path).replace(".jpg", "_depth.png"))
    #     depth_image.save(save_path)

if __name__ == "__main__":
    main()
    

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

# if __name__ == '__main__':
#     # dataset_folder = '/home/chenli/data/UW/SAUD/Benchmark/Best_bt'
#     # SAUD_depth_statistics(dataset_folder)
        
        
#     # images_folder = '/home/chenli/data/images_data/inaturalist_12K/images/train'
#     # depths_folder = '/home/chenli/data/images_data/inaturalist_12K/depths/train'
#     wrong_shape_base_name = [
#         'Insecta_0277', 
#         'Animalia_0544', 
#     ]
    
#     images_folder = '/home/chenli/data/images_data/inaturalist_12K/images/train'
#     depths_folder = '/home/chenli/data/images_data/inaturalist_12K/depths/train'
#     INA12k_depth_pre(images_folder, depths_folder, zoedepth())
    