import os, pdb
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class NYUv2Dataset(Dataset):
    def __init__(self, root_dir, transform=None, depth_mode='Map', depth_path=None, return_image_path = False):
        self.root_dir = root_dir
        self.transform = transform
        self.depth_mode = depth_mode
        self.depth_path = depth_path
        self.return_image_path = return_image_path
        
        # Correctly list images in the 'images' directory
        self.images = os.listdir(os.path.join(self.root_dir, 'images'))
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get image path
        image_path = os.path.join(self.root_dir, 'images', self.images[idx])
        
        # Determine depth path based on depth_mode
        if self.depth_mode == 'Map':
            depth_path = os.path.join(self.root_dir, 'depths', self.images[idx].replace('image', 'depth'))
        elif self.depth_mode == 'Predict' and self.depth_path:
            depth_path = self.depth_path
        else:
            raise NotImplementedError('Only map and predict depth modes are supported.')
        
        # Load image and depth files
        image = np.load(image_path)  
        depth = np.load(depth_path)  # Assuming depth is in .npy format
        # pdb.set_trace()
        # Apply transformations if available
        if self.transform:
            image = self.transform(image)
            # depth = transforms.ToTensor()(depth)  # Convert depth to tensor if necessary

        if self.return_image_path:
            return {'image': image, 'depth': depth, 'image_path': image_path}
        return {'image': image, 'depth': depth}

def get_dataloader(root_dir, return_image_path = False, batch_size=8, shuffle=True, num_workers=4):
    transform = transforms.Compose([
        # transforms.Resize((256, 256)),  # You can change the size as per your requirement
        transforms.ToTensor(),
    ])

    dataset = NYUv2Dataset(root_dir=root_dir, transform=transform, return_image_path=return_image_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return dataloader

if __name__ == '__main__':
    root_dir = '/home/chenli/data/images_data/NYU_v2_label'
    
    nyuv2 = NYUv2Dataset(root_dir)
    # img, depth = nyuv2.__getitem__(0)
    

    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--T_file', required=True, help='Path to template realwater image')
    # parser.add_argument('--air_path', required=True, help='Path to air image folder')
    # parser.add_argument('--depth_path', required=True, help='Path to depth image folder')
    # parser.add_argument('--mode', required=True, help='Mode = {Map, Predict, Hybrid}')
    # args = parser.parse_args()

    # air_image_folder = args.air_path
    # depth_image_folder = args.depth_path

    # dataloader = get_dataloader(air_image_folder, depth_image_folder)

    # for batch_idx, (air_images, depth_images) in enumerate(dataloader):
    #     # You can now process your air images and depth images
    #     print(f"Batch {batch_idx}:")
    #     print(f"Air images shape: {air_images.shape}")
    #     print(f"Depth images shape: {depth_images.shape}")

    #     # Call your generate_underwater_image function here and save the result