import os
import cv2
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
import shutil
import matplotlib.pyplot as plt

# # 图像分类函数，基于 CIELab 和 HSV 空间
# def classify_image(image_path):
#     """
#     根据颜色偏差和饱和度对单张图片进行分类
#     """
#     img = cv2.imread(image_path)
#     if img is None:
#         return "Invalid image"
    
#     # 转换为 CIELab 空间
#     lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
#     l, a, b = cv2.split(lab_img)

#     # 计算 a 和 b 通道的均值
#     a_mean = np.mean(a)
#     b_mean = np.mean(b)
    
#     # 转换为 HSV 空间，计算饱和度均值
#     hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     h, s, v = cv2.split(hsv_img)
#     s_mean = np.mean(s)

#     # 设置颜色分类的阈值
#     green_threshold = -10  # 偏绿检测
#     blue_threshold = 10    # 偏蓝检测
#     yellow_threshold = 10  # 偏黄检测
#     low_saturation_threshold = 50  # 低饱和度检测
    
#     # 分类判断
#     if a_mean < green_threshold:
#         return "偏绿"
#     elif b_mean > blue_threshold:
#         return "偏蓝"
#     elif b_mean < -yellow_threshold:
#         return "偏黄"
#     elif s_mean < low_saturation_threshold:
#         return "低饱和度"
#     else:
#         return "正常"

# # 批量分类数据集中的图片
# def classify_dataset(folder_path):
#     """
#     遍历文件夹中的图片，根据颜色偏差和饱和度进行分类
#     """
#     categories = defaultdict(list)
    
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".jpg") or filename.endswith(".png"):
#             image_path = os.path.join(folder_path, filename)
#             classification = classify_image(image_path)
#             categories[classification].append(filename)
    
#     return categories

# 输出分类结果
def print_classification_results(categories):
    """
    打印分类后的结果
    """
    for category, images in categories.items():
        print(f"分类: {category}, 图片数量: {len(images)}")
        for img in images:
            print(f"  - {img}")

# 计算单张图片的颜色统计数据
def calculate_color_stats(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_hue = np.mean(hsv_image[:, :, 0])
    mean_saturation = np.mean(hsv_image[:, :, 1])
    mean_value = np.mean(hsv_image[:, :, 2])

    return mean_hue, mean_saturation, mean_value

# 处理整个数据集，提取每张图片的颜色统计信息
def process_dataset(image_dir):
    color_stats = []
    image_paths = []
    for image_file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_file)
        if image_file.endswith(('.png', '.jpg', '.jpeg')):
            stats = calculate_color_stats(image_path)
            if stats:
                color_stats.append(stats)
                image_paths.append(image_path)
    return color_stats, image_paths

# 对图像进行聚类
def cluster_images(color_stats, n_clusters=4):
    color_stats_array = np.array(color_stats)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(color_stats_array)
    return labels, kmeans.cluster_centers_

# # 将图像按聚类结果组织到文件夹
# def organize_images_by_cluster(image_paths, labels, output_dir):
#     for idx, image_path in enumerate(image_paths):
#         cluster_folder = os.path.join(output_dir, f'cluster_{labels[idx]}')
#         os.makedirs(cluster_folder, exist_ok=True)
#         shutil.copy2(image_path, os.path.join(cluster_folder, os.path.basename(image_path)))

# # 可视化聚类中心的颜色统计信息
# def plot_cluster_color_stats(kmeans, cluster_centers):
#     fig, ax = plt.subplots(figsize=(10, 6))
#     for i in range(len(cluster_centers)):
#         hue, saturation, value = cluster_centers[i]
#         rgb_color = mcolors.hsv_to_rgb([hue / 179, saturation / 255, value / 255])
#         ax.bar(i, value, color=rgb_color, label=f'Cluster {i+1}')
    
#     ax.set_xlabel('Cluster')
#     ax.set_ylabel('Mean Value')
#     ax.set_title('Cluster Centers Color Stats (HSV to RGB)')
#     ax.legend()

#     plt.savefig('./cluster_color_stats_rgb.png', dpi=300, bbox_inches='tight')
    
#     # 图像分类函数，基于 CIELab 和 HSV 空间
# def classify_image(image_path):
#     """
#     根据颜色偏差和饱和度对单张图片进行分类
#     """
#     img = cv2.imread(image_path)
#     if img is None:
#         return "Invalid image"
    
#     # 转换为 CIELab 空间
#     lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
#     l, a, b = cv2.split(lab_img)

#     # 计算 a 和 b 通道的均值
#     a_mean = np.mean(a)
#     b_mean = np.mean(b)
    
#     # 转换为 HSV 空间，计算饱和度均值
#     hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     h, s, v = cv2.split(hsv_img)
#     s_mean = np.mean(s)

#     # 设置颜色分类的阈值
#     green_threshold = -10  # 偏绿检测
#     blue_threshold = 10    # 偏蓝检测
#     yellow_threshold = 10  # 偏黄检测
#     low_saturation_threshold = 50  # 低饱和度检测
    
#     # 分类判断
#     if a_mean < green_threshold:
#         return "偏绿"
#     elif b_mean > blue_threshold:
#         return "偏蓝"
#     elif b_mean < -yellow_threshold:
#         return "偏黄"
#     elif s_mean < low_saturation_threshold:
#         return "低饱和度"
#     else:
#         return "正常"

# # 批量分类数据集中的图片
# def classify_dataset(folder_path):
#     """
#     遍历文件夹中的图片，根据颜色偏差和饱和度进行分类
#     """
#     categories = defaultdict(list)
    
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".jpg") or filename.endswith(".png"):
#             image_path = os.path.join(folder_path, filename)
#             classification = classify_image(image_path)
#             categories[classification].append(image_path)
    
#     return categories

# 将图像按分类结果组织到文件夹
def organize_images_by_category(categories, output_dir):
    """
    根据分类结果将图像组织到文件夹
    """
    for category, image_paths in categories.items():
        category_folder = os.path.join(output_dir, category)
        os.makedirs(category_folder, exist_ok=True)
        for image_path in image_paths:
            # image_path = os.path.join(input_dir, image_path)
            shutil.copy2(image_path, os.path.join(category_folder, os.path.basename(image_path)))

# 可视化分类结果的颜色统计信息
def plot_category_color_stats(categories):
    """
    根据分类结果计算每个类别的平均颜色，并进行可视化
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (category, image_paths) in enumerate(categories.items()):
        total_hue, total_saturation, total_value, count = 0, 0, 0, 0
        for image_path in image_paths:
            img = cv2.imread(image_path)
            if img is None:
                continue
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv_img)
            total_hue += np.mean(h)
            total_saturation += np.mean(s)
            total_value += np.mean(v)
            count += 1

        if count > 0:
            avg_hue = total_hue / count
            avg_saturation = total_saturation / count
            avg_value = total_value / count
            rgb_color = mcolors.hsv_to_rgb([avg_hue / 179, avg_saturation / 255, avg_value / 255])
            ax.bar(i, avg_value, color=rgb_color, label=category)
    
    ax.set_xlabel('Category')
    ax.set_ylabel('Mean Value')
    ax.set_title('Category Color Stats (HSV to RGB)')
    ax.legend()
    
    plt.savefig('category_color_stats_rgb.png', dpi=300, bbox_inches='tight')
    
# 图像分类函数，基于 CIELab 和 HSV 空间
def classify_image(image_path):
    """
    根据颜色偏差和饱和度对单张图片进行宽松分类
    """
    img = cv2.imread(image_path)
    if img is None:
        return "Invalid image"
    
    # 转换为 CIELab 空间
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab_img)

    # 计算 a 和 b 通道的均值
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    
    # 转换为 HSV 空间，计算饱和度均值
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    s_mean = np.mean(s)

    # 调整后的阈值，先宽松分类
    green_threshold = -5    # 偏绿检测
    blue_threshold = 5      # 偏蓝检测
    yellow_threshold = 5    # 偏黄检测
    low_saturation_threshold = 60  # 低饱和度检测
    
    # 分类判断（宽松版）
    if a_mean < green_threshold:
        return "偏绿"
    elif b_mean > blue_threshold:
        return "偏蓝"
    elif b_mean < -yellow_threshold:
        return "偏黄"
    elif s_mean < low_saturation_threshold:
        return "低饱和度"
    else:
        return "正常"

# 批量分类数据集中的图片
def classify_dataset(folder_path):
    """
    遍历文件夹中的图片，进行宽松分类，并从中挑选代表性图像
    """
    categories = defaultdict(list)
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            classification = classify_image(image_path)
            categories[classification].append(image_path)
    
    # Refine the classification by further selecting representative images (if needed)
    refined_categories = refine_classification(categories)
    
    return refined_categories

# 进一步精细化分类结果
def refine_classification(categories):
    """
    根据每个类别的图像数量，对每个类别选择一部分代表性图像
    """
    refined_categories = defaultdict(list)
    for category, images in categories.items():
        if len(images) > 10:  # 如果该类有超过 10 张图片，选择 10 张作为代表
            refined_categories[category] = images[:10]
        else:
            refined_categories[category] = images
    return refined_categories

# 输出分类结果
def print_classification_results(categories):
    """
    打印分类后的结果
    """
    for category, images in categories.items():
        print(f"分类: {category}, 图片数量: {len(images)}")
        for img in images:
            print(f"  - {img}")
            

# 计算整个数据集中 CIELab 和 HSV 空间的统计信息
def compute_global_stats(folder_path):
    a_values = []
    b_values = []
    s_values = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            img = cv2.imread(image_path)
            if img is not None:
                # 转换为 CIELab 空间
                lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
                l, a, b = cv2.split(lab_img)
                
                # 转换为 HSV 空间，计算饱和度
                hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv_img)

                # 记录 a、b 通道和饱和度的均值
                a_values.append(np.mean(a))
                b_values.append(np.mean(b))
                s_values.append(np.mean(s))

    # 计算全局均值和标准差
    a_mean = np.mean(a_values)
    a_std = np.std(a_values)
    b_mean = np.mean(b_values)
    b_std = np.std(b_values)
    s_mean = np.mean(s_values)
    s_std = np.std(s_values)

    return a_mean, a_std, b_mean, b_std, s_mean, s_std

# 动态设置阈值进行分类
def classify_image_dynamic(image_path, a_mean, a_std, b_mean, b_std, s_mean, s_std):
    img = cv2.imread(image_path)
    if img is None:
        return "Invalid image"
    
    # 转换为 CIELab 空间
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab_img)

    # 计算 a 和 b 通道的均值
    a_img_mean = np.mean(a)
    b_img_mean = np.mean(b)
    
    # 转换为 HSV 空间，计算饱和度均值
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    s_img_mean = np.mean(s)

    # 动态分类标准，基于全局统计信息
    green_threshold = a_mean - a_std  # 偏绿
    blue_threshold = b_mean + b_std   # 偏蓝
    yellow_threshold = b_mean - b_std # 偏黄
    low_saturation_threshold = s_mean - s_std  # 低饱和度
    
    # 分类判断
    if a_img_mean < green_threshold:
        return "偏绿"
    elif b_img_mean > blue_threshold:
        return "偏蓝"
    elif b_img_mean < yellow_threshold:
        return "偏黄"
    elif s_img_mean < low_saturation_threshold:
        return "低饱和度"
    else:
        return "正常"

# 批量分类数据集中的图片
def classify_dataset_dynamic(folder_path):
    # 计算全局统计信息
    a_mean, a_std, b_mean, b_std, s_mean, s_std = compute_global_stats(folder_path)
    
    categories = defaultdict(list)
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            classification = classify_image_dynamic(image_path, a_mean, a_std, b_mean, b_std, s_mean, s_std)
            categories[classification].append(image_path)
    
    return categories

# # 示例：对整个数据集进行分类
# folder_path = "path_to_your_image_dataset"
# categories = classify_dataset_dynamic(folder_path)
# print_classification_results(categories)
