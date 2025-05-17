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
import sys, os, re, pdb

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
from skimage import exposure
from skimage.restoration import denoise_bilateral
# import matplotlib

from midas_helper import run_midas

# current_dir = os.path.dirname(os.path.abspath(__file__))
# so_path = os.path.join(current_dir, 'sillu.so')

# lib = np.ctypeslib.load_library('sillu','.')
# 获取 a.py 所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 加载 .so 文件
lib = np.ctypeslib.load_library('sillu', current_dir)
# lib = np.ctypeslib.load_library('comllu', '.')

# matplotlib.use('TkAgg')

NUM_BINS = 10 # number of bins of depths to find backscatter

################################ File operations ###############################
def read_folder_file(folder):
    files = [os.path.join(folder, file) for file in os.listdir(folder)]
    return files

def read_image(image_path, max_side = 3840):
    if image_path.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
        image_file = Image.open(image_path)
        image_file = image_file.convert('RGB')
    else:
        image_file_raw = rawpy.imread(image_path).postprocess()
        image_file = Image.fromarray(image_file_raw)
    image_file.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return np.float64(image_file) / 255.0

def read_depthmap(depthmap_path, size):
    depth_file = Image.open(depthmap_path)
    depths = depth_file.resize(size, Image.Resampling.LANCZOS)
    return np.float64(depths)

# pfm 文件 read_depthmap 虽然能读取，但存在问题
def read_pfm(path):
    """Read pfm file.

    Args:
        path (str): path to file

    Returns:
        tuple: (data, scale)
    """
    with open(path, "rb") as file:

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file: " + path)

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception("Malformed PFM header.")

        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:
            # little-endian
            endian = "<"
            scale = -scale
        else:
            # big-endian
            endian = ">"

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data, scale

def rawfile_path2related_path(raw_file_path, depth_or_params):
    image_dir = os.path.dirname(raw_file_path)
    image_name = os.path.splitext(os.path.basename(raw_file_path))[0]
        
    related_dir = os.path.join(image_dir, '../' + depth_or_params)
    related_dir = os.path.abspath(related_dir)
    
    if not os.path.exists(related_dir):
        os.makedirs(related_dir)
        
    if depth_or_params == 'depth':
        related_name = f'depth{image_name}.tif'
    elif depth_or_params == 'params' or depth_or_params =='params_zoe' or depth_or_params =='params_zoe_norm':
        related_name = f'{image_name}.pkl'
    elif depth_or_params == 'sea_thru' or depth_or_params == 'sea_thru_zoe' or depth_or_params =='sea_thru_zoe_norm':
        related_name = f'{image_name}.png'
    else:
        raise ValueError("Invalid depth_or_params. It should be 'depth', 'sea_thru', or 'params'")
    related_path = os.path.join(related_dir, related_name)
    
    return related_path
    

def read_raw_depth_file(raw_file_path, depth_path = None, mode='Map', hint = None, size = None, prefix = None):
    
    if size is not None:
        original = read_image(raw_file_path, size)
    else:
        original = read_image(raw_file_path)
    
    depth_path = rawfile_path2related_path(raw_file_path, 'depth')
    
    if mode == "Map":
        # Using given depth map
        print("Using user input depth map")
        depths = read_depthmap(depth_path, (original.shape[1], original.shape[0]))
        depths = normalize_depth_map(depths, 0.1, 6.0)
    elif mode == "Predict":
        # Predicting depth using MiDaS
        print("Predicting depth using MiDaS")
        depths = run_midas(original, "out/", "weights/dpt_large-midas-2f21e586.pt", "dpt_large")
        # depths = run_midas(args.original, "out/", "weights/dpt_hybrid-midas-501f0c75.pt", "dpt_hybrid")
        depths = cv2.resize(depths, dsize = (original.shape[1], original.shape[0]), interpolation = cv2.INTER_CUBIC)
        # depths = np.square(depths) # More contrast!
        depths = np.max(depths) / depths # disparity map to depth map
        print(depths)

        if hint is not None:
            print("Preprocessing monocular depths esimation with hint")    
            hint_depths = read_depthmap(hint, (original.shape[1], original.shape[0]))
            depths = refine_depths_from_hint(depths, np.mean(hint_depths))
        else:
            print("Preprocessing monocular depths esimation without hint")    
            preprocess_predicted_depths(original, depths)
    elif mode == "Hybrid":
        print("Loading user input depth map")
        depths = read_depthmap(depth_path, (original.shape[1], original.shape[0]))
        print("Predicting depth using MiDaS")
        pdepths = run_midas(original, "out/", "weights/dpt_large-midas-2f21e586.pt", "dpt_large")
        pdepths = cv2.resize(pdepths, dsize = (original.shape[1], original.shape[0]), interpolation = cv2.INTER_CUBIC)
        pdepths = np.max(pdepths) / pdepths # disparity map to depth map
        print("Combining depth maps")
        depths = combine_map_predict(depths, pdepths)
        
    return original, depths

def save_params(dict, raw_file_path, child_folder):
    pkl_path = rawfile_path2related_path(raw_file_path, child_folder)
    with open(pkl_path, 'wb') as f:
        pickle.dump(dict, f)

    print(f"Parameters saved to {pkl_path}")

################################ Preprocessing ############################
def normalize_depth_map(depths, z_min, z_inf):
    """
    Normalize values in the depth map
    """
    z_max = max(np.max(depths), z_inf)
    depths[depths == 0] = z_max
    depths[depths < z_min] = z_min
    return depths

def estimate_far(image, frac = 0.2, close = 0.3):
    """
    Estimates the farthest distance from the image color    
    """
    # Chosen luminance formula
    r = image[:, :, 0] * 0.2126
    g = image[:, :, 1] * 0.7152
    b = image[:, :, 2] * 0.0722

    lum = np.sum(np.stack([r, g, b], axis = 2), axis = 2)
    lum.sort(axis = 0)
    # print(lum)

    rows = int(frac * lum.shape[0])

    darkest = np.mean(lum[rows:(2 * rows), :], axis = 0)
    brightest = np.mean(lum[-(2 * rows):(-rows), :], axis = 0)

    ratio = np.mean(brightest / darkest)

    return np.log2(ratio) * 10

def refine_depths_from_hint(depths, avg):
    oavg = np.mean(depths)
    depths = depths * (avg / oavg)
    print("Estimated farthest distance is {far}".format(far = np.max(depths)))
    print("Estimated mean distance is {mean}".format(mean = np.mean(depths)))
    return depths

def preprocess_predicted_depths(original, depth):
    far = estimate_far(original)
    print("Estimated farthest distance is {far}".format(far = far))
    # print(np.min(depth), np.max(depth), np.isnan(depth).sum(), np.isinf(depth).sum(), np.any(depth<=0)) 
    ratio = far / np.max(depth)
    # depth = np.nan_to_num(depth, nan=np.finfo(np.float32).eps, posinf=np.max(depth), neginf=np.min(depth))
    print("Estimated mean distance is {mean}".format(mean = np.mean(depth * ratio)))
    return depth * ratio

def combine_map_predict(depths, predict):
    idxs = np.where(depths == 0)
    mfar = np.percentile(depths, 95)
    far_idxs = np.where(depths > mfar)
    pfar = np.min(predict[far_idxs])
    ratio = mfar / pfar
    depths[idxs] = predict[idxs] * ratio
    depths = denoise_bilateral(depths)
    return depths

################################## Backscatter ############################

# Eq. 10
def predict_backscatter(z, veil, backscatter, recover, attenuation):
    return (veil * (1 - np.exp(-backscatter * z)) + recover * np.exp(-attenuation * z))

def find_reference_points(image, depths, frac = 0.02):
    z_max = np.max(depths)
    z_min = np.min(depths)
    z_bins = np.linspace(z_min, z_max, NUM_BINS + 1)
    rgb_norm = np.linalg.norm(image, axis=2) # is using 2-norm correct here?
    ret = []
    for i in range(NUM_BINS):
        lo, hi = z_bins[i], z_bins[i + 1]
        indices = np.where(np.logical_and(depths >= lo, depths < hi))
        if indices[0].size == 0:
            continue
        bin_rgb_norm, bin_z, bin_color = rgb_norm[indices], depths[indices], image[indices]
        points_sorted = sorted(zip(bin_rgb_norm, bin_z, bin_color[:,0], bin_color[:,1], bin_color[:,2]), key = lambda p : p[0])
        for j in range(math.ceil(len(points_sorted) * frac)):
            ret.append(points_sorted[j])
    return np.asarray(ret)

def estimate_channel_backscatter(points, depths, channel, attempts = 50):
    lo = np.array([0, 0, 0, 0])
    hi = np.array([1, 5, 1, 5])
    # hi = np.array([2, 10, 2, 10])

    best_loss = np.inf
    best_coeffs = []

    for _ in range(attempts):
        try:
            popt, pcov = scipy.optimize.curve_fit(predict_backscatter, points[:, 1], points[:, channel + 2], 
                                                np.random.random(4) * (hi - lo) + lo, bounds = (lo, hi))
            cur_loss = np.mean(np.square(predict_backscatter(points[:, 1], *popt) - points[:, channel + 2]))
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_coeffs = popt
        except RuntimeError as re:
            print(re, file=sys.stderr)

    print("Found coeffs for channel {channel} with mse {mse}".format(channel = channel, mse = best_loss))
    print("Veil = {Veil}, backscatter = {backscatter}, recover = {recover}, attenuation = {attenuation}".format(
        Veil = best_coeffs[0], backscatter = best_coeffs[1], recover = best_coeffs[2], attenuation = best_coeffs[3]))

    Bc_channel = predict_backscatter(depths, *best_coeffs)

    return Bc_channel, best_coeffs

def estimate_backscatter(image, depths):
    points = find_reference_points(image, depths)
    backscatter_channels = []
    backscatter_coeffs = []
    for channel in range(3):
        Bc, coeffs = estimate_channel_backscatter(points, depths, channel)
        backscatter_channels.append(Bc)
        backscatter_coeffs.append(coeffs)

    Ba = np.stack(backscatter_channels, axis = 2)

    return Ba, backscatter_coeffs

# Wideband attenuation

def predict_wideband_attenuation(depths, a, b, c, d):
    return a * np.exp(b * depths) + c * np.exp(d * depths)

def predict_z(x, a, b, c, d):
    Ec, depth = x
    return -np.log(Ec) / (a * np.exp(b * depth) + c * np.exp(d * depth))

def find_reference_points_att(image, depths, channel, frac = 0.02):
    z_max = np.max(depths)
    z_min = np.min(depths)
    z_bins = np.linspace(z_min, z_max, NUM_BINS + 1)
    ret = []
    for i in range(NUM_BINS):
        lo, hi = z_bins[i], z_bins[i + 1]
        indices = np.where(np.logical_and(depths >= lo, depths < hi))
        if indices[0].size == 0:
            continue
        bin_z, bin_color = depths[indices], image[indices]
        points_sorted = sorted(zip(bin_z, bin_color[:, channel]), key = lambda p : -p[1])
        for j in range(math.ceil(len(points_sorted) * frac)):
            ret.append(points_sorted[j])
    return np.asarray(ret)

def estimate_wideband_attenuation(D, depths, coarse_attempts = 5):
    """
    Args:
        Dc: direct signal
    """
    # Ea = compute_illuminant_map_plugin(D, depths, p = 0.5, f = 2.0, eps = 0.03)
    Ea = compute_illuminant_map_plugin(D, depths, p = 0.4, f = 2.0, eps = np.mean(depths) / 50.0)
    Ea = np.clip(Ea, 0, 1)

    att_channels = []
    coeffs_D = []

    for channel in range(3):
        # points = find_reference_points_att(Ea, depths, channel)
        # depths_flatten = points[:, 0]
        # Ec = points[:, 1]
        Ec = np.copy(Ea[:, :, channel])
        Ec.reshape(-1)
        depths_flatten = np.copy(depths)
        depths_flatten.reshape(-1)

        locs = np.where(Ec > 1E-5)
        Ec = Ec[locs]
        depths_flatten = depths_flatten[locs]

        # Coarse estimate
        beta_D_hat = -np.log(Ec) / depths_flatten

        lo = np.array([0, -10, 0, -10])
        hi = np.array([100, 0, 100, 0])

        b_lo = np.array([0, -np.inf, 0, -np.inf])
        b_hi = np.array([np.inf, 0, np.inf, 0])

        best_loss = np.inf
        best_coeffs = []
        for _ in range(coarse_attempts):
            try:
                popt, pcov = scipy.optimize.curve_fit(predict_wideband_attenuation,
                                                    depths_flatten, beta_D_hat, 
                                                    np.random.random(4) * (hi - lo) + lo, bounds = (b_lo, b_hi))
                cur_loss = np.mean(np.square(predict_z((Ec, depths_flatten), *popt) - depths_flatten))
                if cur_loss < best_loss:
                    best_loss = cur_loss
                    best_coeffs = popt
            except RuntimeError as re:
                print(re, file=sys.stderr)
        
        print("Coarse estimate gives coeffs for channel {channel} with mse {mse}".format(channel = channel, mse = best_loss))
        print("a = {a}, b = {b}, c = {c}, d = {d}".format(
            a = best_coeffs[0], b = best_coeffs[1], c = best_coeffs[2], d = best_coeffs[3]))

        coeffs = refine_attenuation_estimation(Ec, depths_flatten, channel, best_coeffs)
        coeffs_D.append(coeffs)
        att_channel = predict_wideband_attenuation(depths, *coeffs)
        att_channels.append(att_channel)
    
    att = np.stack(att_channels, axis = 2)
    
    return att, Ea, coeffs_D

def refine_attenuation_estimation(Ec, depths, channel, x0, attempts = 5):
    """
    Ec is illuminant map of only 1 channel
    """
    # Curve fitting
    lo = np.array([0, -10, 0, -10])
    hi = np.array([100, 0, 100, 0])

    b_lo = np.array([0, -np.inf, 0, -np.inf])
    b_hi = np.array([np.inf, 0, np.inf, 0])

    best_loss = np.inf
    best_coeffs = []

    original_shape = depths.shape
    Ec.reshape(-1)
    depths.reshape(-1)

    locs = np.where((Ec > 1E-5) & (depths < np.percentile(depths, 95)) & (depths > np.percentile(depths, 10)))
    E = Ec[locs]
    z = depths[locs]

    try:
        popt, pcov = scipy.optimize.curve_fit(predict_z,
                                            (E, z), z, 
                                            x0, bounds = (b_lo, b_hi))
        cur_loss = np.mean(np.square(predict_z((E, z), *popt) - z))
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_coeffs = popt
    except RuntimeError as re:
        print(re, file=sys.stderr)

    print("Found coeffs for channel {channel} with mse {mse}".format(channel = channel, mse = best_loss))
    print("a = {a}, b = {b}, c = {c}, d = {d}".format(
        a = best_coeffs[0], b = best_coeffs[1], c = best_coeffs[2], d = best_coeffs[3]))

    depths.reshape(original_shape)

    return best_coeffs

def compute_illuminant_map_plugin(D, depths, iterations = 500, p = 0.5, f = 2, eps = 0.2):
    """
    Calls C interface for computing illuminant map and returns LSAC result
    """
    func = lib.compute_illuminant_map

    func.restype = None
    func.argtypes = [np.ctypeslib.ndpointer(float, ndim = 2, flags = 'aligned, contiguous'),
                     np.ctypeslib.ndpointer(float, ndim = 2, flags = 'aligned, contiguous'),
                     np.ctypeslib.ndpointer(float, ndim = 2, flags = 'aligned, contiguous, writeable'),
                     ctypes.c_double,
                     ctypes.c_double,
                     ctypes.c_double,
                     ctypes.c_int, 
                     ctypes.c_int,
                     ctypes.c_int]

    a = []

    z = np.require(depths, float, ['ALIGNED', 'CONTIGUOUS'])

    for channel in range(3):
        Dc = np.ascontiguousarray(D[:, :, channel])
        Dc = np.require(Dc, float, ['ALIGNED', 'CONTIGUOUS'])
        ac = np.zeros_like(Dc)
        ac = np.require(ac, float, ['ALIGNED', 'CONTIGUOUS'])
        x, y = depths.shape
        func(Dc, z, ac, p, f, eps, x, y, iterations, dtype = float)
        a.append(ac)

    return np.stack(a, axis = 2)

def compute_illuminant_map(Dc, depths, iterations = 100, p = 0.5, f = 2, eps = 0.03):
    """
    Computes illuminant map and returns LSAC result, very slow, only provided as a reference implementation
    """
    neighborhood_maps = compute_neighborhood(depths)

    ac = np.zeros_like(Dc)
    ac_p = ac.copy()
    ac_new = ac.copy()

    xlim, ylim, _ = Dc.shape

    for _ in range(iterations):
        for x in range(xlim):
            for y in range(ylim):
                idcs = neighborhood_maps[x][y]
                ac_p[x, y] = np.sum(ac[tuple(idcs)], axis = 0) / len(idcs[0])
        ac_new = Dc * p + ac_p * (1 - p)
        if np.max(np.abs(ac - ac_new)) < eps:
            break
        ac = ac_new

    return ac * f

def compute_neighborhood(depths, epsilon = 0.03):
    """
    This could be very expensive...
    """
    
    xlim, ylim = depths.shape

    neighborhood_maps = []
    for x in range(xlim):
        print("Process: {p}".format(p = x / xlim))
        row = []
        for y in range(ylim):
            row.append(find_neighborhood(depths, x, y, epsilon))
        neighborhood_maps.append(row)
    
    return neighborhood_maps

def find_neighborhood(depths, x, y, epsilon):
    flags = np.zeros_like(depths, dtype = np.intc)
    xlim, ylim = depths.shape

    z = depths.copy()
    z = np.abs(z - z[x, y])

    q = queue.Queue()
    q.put((x, y))

    ret = [[], []]

    while not q.empty():
        cur_x, cur_y = q.get()
        if flags[cur_x, cur_y]:
            continue
        flags[cur_x, cur_y] = 1
        if z[cur_x, cur_y] < epsilon:
            ret[0].append(cur_x)
            ret[1].append(cur_y)
            if cur_x > 0:
                q.put((cur_x - 1, cur_y))
            if cur_y > 0:
                q.put((cur_x, cur_y - 1))
            if cur_x < xlim - 1: 
                q.put((cur_x + 1, cur_y))
            if cur_y < ylim - 1:
                q.put((cur_x, cur_y + 1))

    return ret

def recover(Da, att, depths):
    for c in range(3):
        att[:, :, c] = att[:, :, c] * depths

    Ja = Da * np.exp(att)

    Ja = np.clip(Ja, 0, 1)
    # Ja = Ja / Ja.max()

    return Ja

# Whitebalance

# Whitebalancing with 
# Conversion functions courtesy of https://stackoverflow.com/a/34913974/2721685
def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:, :, [1,2]] += 128
    return ycbcr #np.uint8(ycbcr)

def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float64)
    rgb[:, :, [1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)

def wb_ycbcr_mean(data):
    # Convert data and sample to YCbCr
    ycbcr = rgb2ycbcr(data)

    # Calculate mean components
    yc = list(np.mean(ycbcr[:, :, i]) for i in range(3))

    # Center cb and cr components of image based on sample
    for i in range(1,3):
        ycbcr[:, :, i] = np.clip(ycbcr[:, :, i] + (128 - yc[i]), 0, 255)
    
    return ycbcr2rgb(ycbcr)

def grey_world(img):
    dg = 1.0 / np.mean(np.sort(img[:, :, 1], axis=None)[int(round(-1 * np.size(img[:, :, 0]) * 0.1)):])
    db = 1.0 / np.mean(np.sort(img[:, :, 2], axis=None)[int(round(-1 * np.size(img[:, :, 0]) * 0.1)):])
    dsum = dg + db
    dg = dg / dsum * 2.
    db = db / dsum * 2.
    img[:, :, 0] *= (db + dg) / 2
    img[:, :, 1] *= dg
    img[:, :, 2] *= db
    return img



############################# forward #######################################

def compute_forward_backscatter(depths, coeffs_B):
    """
    Calculate backscatter component for the forward imaging process.

    Parameters:
    depths (numpy array): Depth values corresponding to each pixel.
    coeffs (list of arrays): Coefficients for each channel [veil, backscatter, recover, attenuation].

    Returns:
    Bc_forward (numpy array): Calculated background scattering B_c for each channel.
    """
    
    # Initialize an empty list to store the background scattering for each channel
    Bc_channels = []
    
    # Loop through each color channel (Red, Green, Blue)
    for channel_coeffs in coeffs_B:
        veil = channel_coeffs[0]  # B_c^infinity
        backscatter = channel_coeffs[1]  # beta_c^B
        recover = channel_coeffs[2]  # B_c^infinity
        att = channel_coeffs[3]  # beta_c^B
        
        # Calculate B_c using the formula B_c = veil * (1 - exp(-backscatter * z))
        Bc = veil * (1 - np.exp(-backscatter * depths))
        # Bc = predict_backscatter(depths, veil, backscatter, recover, att)
        
        # Append the result for this channel
        Bc_channels.append(Bc)
    
    # Stack the results along the color channel axis (depths shape should match image dimensions)
    Bc_forward = np.stack(Bc_channels, axis=2)  # Shape: (H, W, 3) for RGB channels
    
    return Bc_forward


def compute_forward_direct_signal(depths, Jc, coeffs_D):
    """
    Compute the direct signal D_c in the forward underwater imaging process.

    Parameters:
    depths (numpy array): Depth values for each pixel.
    Jc (numpy array): Unattenuated scene radiance for each pixel (shape: H, W, 3).
    coeffs_D (list of arrays): Coefficients [a, b, c, d] for each channel (length 3 for RGB).

    Returns:
    Dc (numpy array): Calculated direct signal D_c for each channel (shape: H, W, 3).
    """
    
    # Initialize an empty list to store the direct signal for each channel
    Dc_channels = []
    
    # Loop through each color channel (Red, Green, Blue)
    for channel in range(3):  # Iterate over the three channels
        a, b, c, d = coeffs_D[channel]  # Coefficients for the current channel
        # Calculate beta_c^D(z) using the formula beta_c^D(z) = ae^{bz} + ce^{dz}
        beta_D = predict_wideband_attenuation(depths, a, b, c, d)
        
        # Calculate the direct signal D_c = J_c * exp(-beta_c^D(z) * z) for the current channel
        Dc = Jc[:, :, channel] * np.exp(-beta_D * depths)
        
        # Append the result for this channel
        Dc_channels.append(Dc)
    
    # Stack the results along the color channel axis (depths shape should match image dimensions)
    Dc_forward = np.stack(Dc_channels, axis=2)  # Shape: (H, W, 3) for RGB channels
    
    return Dc_forward
def generate_underwater_image(air, depths, coeffs_B, coeffs_D, post_proc = 'clip'):
    Bc_forward = compute_forward_backscatter(depths, coeffs_B)
    # Bc_forward_ = Image.fromarray((Bc_forward * 255).astype(np.uint8))
    # Bc_forward_.save("out/" + prefix + "Bc_forward.png")
    
    Dc_forward = compute_forward_direct_signal(depths, air, coeffs_D)
    # Dc_forward_ = Image.fromarray((Dc_forward * 255).astype(np.uint8))
    # Dc_forward_.save("out/" + prefix + "Dc_forward.png")

    # Generate the final underwater image by combining the simulated direct signal and background scattering
    underwater_image = Bc_forward + Dc_forward

    # Clip the values to the valid range [0, 1] and convert back to 8-bit format
    if post_proc == 'clip':
        underwater_image = np.clip(underwater_image, 0, 1)
    elif post_proc == 'norm':
        underwater_image /= underwater_image.max()
    else:
        raise NotImplementedError(f'{post_proc} not implemented')
    # underwater_image = np.uint8(underwater_image * 255.0)
    
    return underwater_image
    

    
def load_template_params(raw_file_path, mode, hint, size, prefix):
    '''
        1. 读取original 及 depth
        2. 有params文件
            1. 读取params, (Ba, coeffs, att, Ea)
            2. 计算Da
                1. Da = original - Ba
        3. 无params文件
            1. 计算Ba, coeffs
            2. 计算Da
            3. 计算att, Ea
            4. 保存params, (Ba, coeffs, att, Ea)
    '''
    original, depths = read_raw_depth_file(raw_file_path = raw_file_path, mode = mode, hint = hint, size = size, prefix = prefix)
        
    # if params file exists
    pkl_path = rawfile_path2related_path(raw_file_path, 'params')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            params = pickle.load(f)
            Ba = params['Ba']
            coeffs_B = params['coeffs_B']
            coeffs_D = params['coeffs_D']
            att = params['att']
            Ea = params['Ea']
        Da = original - Ba
        Da = np.clip(Da, 0, 1)
        
        D = np.uint8(Da * 255.0)
        backscatter_removed = Image.fromarray(D)
        backscatter_removed.save("out/" + prefix + "direct_signal.png")
        
    else:
        print("Estimating backscatter...")
        Ba, coeffs_B = estimate_backscatter(original, depths)

        Da = original - Ba
        Da = np.clip(Da, 0, 1)

        D = np.uint8(Da * 255.0)
        backscatter_removed = Image.fromarray(D)
        backscatter_removed.save("out/" + prefix + "direct_signal.png")

        print("Estimating wideband attenuation...")
        att, Ea, coeffs_D = estimate_wideband_attenuation(Da, depths)
        params = {'Ba': Ba,
                  'coeffs_B': coeffs_B,
                  'coeffs_D': coeffs_D,
                  'att': att,
                  'Ea': Ea}
        save_params(params, raw_file_path)
    
    return Ba, coeffs_B, coeffs_D, att, Ea, Da, depths

def calulate_Ja(raw_file_path, mode='Map', hint = None, size = None, prefix = None):
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
    
def add_water(raw_file_path, template_file_path = None, mode='Map', hint = None, size = None, prefix = None):
    '''
        1. 读入图片
        2. 读入 or predict depth
        3. 读入 params
    '''
    prefix = (prefix if (prefix is not None) else "")
    # 读入图片
    original, depths = read_raw_depth_file(raw_file_path = raw_file_path, mode = mode, hint = hint, size = size, prefix = prefix)
    
    # 读入 params
    template_file_path = raw_file_path
    Ba_t, coeffs_B_t, coeffs_D_t, att_t, Ea_t, Da_t, depths_t = load_template_params(template_file_path, mode, hint, size, prefix)
    # Ba_t_ = Image.fromarray((Ba_t * 255).astype(np.uint8))
    # Ba_t_.save("out/" + prefix + "Ba_t.png")

    Ja = recover(Da_t, att_t, depths_t)
    # Ja = Ja / np.max(Ja)
    # Ja = exposure.equalize_adapthist(Ja)

    # Ja *= 255.0
    # # Js = np.uint8(Ja)
    # Ja = wb_ycbcr_mean(Ja)
    # Ja = Ja / np.max(Ja)
    # add wather

    water_image = generate_underwater_image(Ja, depths, coeffs_B_t, coeffs_D_t, prefix)
    water_image_ = Image.fromarray(water_image)
    water_image_.save("out/" + prefix + "water_image.png")
    
    # adjust brghtness
    # water_image_ad = adjust_brightness_to_match(original, water_image)

    # water_iamge = water_iamge / np.max(water_iamge)
    # water_iamge = exposure.equalize_adapthist(water_iamge)
    # water_iamge *= 255.0
    # water_iamge = np.uint8(water_iamge)
    # water_iamge = wb_ycbcr_mean(water_iamge)
    # pdb.set_trace()
    # water_image_ad = Image.fromarray(water_image_ad)
    # water_image_ad.save("out/" + prefix + "water_image_ad.png")
def read_tem_params(template_path):
    with open(template_path, 'rb') as f:
        params = pickle.load(f)
        # Ba = params['Ba']
        coeffs_B = params['coeffs_B']
        coeffs_D = params['coeffs_D']
        
    return coeffs_B, coeffs_D
            # att = params['att']
            # Ea = params['Ea']
        # Da = original - Ba
        # Da = np.clip(Da, 0, 1)
        
        # D = np.uint8(Da * 255.0)
# def add_water_from_tem(image, depth, template_file_path):
#     '''
#         1. 读入图片
#         2. 读入 or predict depth
#         3. 读入 params
#     '''
#     coeffs_B, coeffs_D = read_tem_params(template_file_path)
    
#     water_image = generate_underwater_image(image, depth, coeffs_B, coeffs_D)

#     # water_image_ = Image.fromarray(water_image)
#     return water_image

# def adjust_brightness_to_match(original_image, water_image):
#     """
#     Adjusts the brightness of the water_image to match the average brightness of the original_image.

#     Parameters:
#     original_image (PIL.Image): The original image.
#     water_image (PIL.Image): The water image that needs brightness adjustment.

#     Returns:
#     adjusted_image (PIL.Image): The brightness-adjusted water image.
#     """
#     original_image = Image.fromarray((original_image * 255).astype(np.uint8))
#     water_image = Image.fromarray(water_image)
#     # Convert both images to YCbCr color space
#     original_ycbcr = original_image.convert('YCbCr')
#     water_ycbcr = water_image.convert('YCbCr')

#     # Extract Y channel (brightness)
#     original_y, _, _ = original_ycbcr.split()
#     water_y, cb, cr = water_ycbcr.split()

#     # Convert to numpy arrays for easier manipulation
#     original_y = np.asarray(original_y, dtype=np.float32)
#     water_y = np.asarray(water_y, dtype=np.float32)

#     # Compute mean brightness of both images
#     original_brightness_mean = np.mean(original_y)
#     water_brightness_mean = np.mean(water_y)

#     # Calculate the brightness difference
#     brightness_adjustment_factor = original_brightness_mean / water_brightness_mean

#     # Adjust the brightness of the water image
#     adjusted_y = np.clip(water_y * brightness_adjustment_factor, 0, 255).astype(np.uint8)

#     # Recombine the adjusted Y channel with original Cb and Cr channels
#     adjusted_ycbcr = Image.merge('YCbCr', (Image.fromarray(adjusted_y), cb, cr))

#     # Convert back to RGB
#     adjusted_image = adjusted_ycbcr.convert('RGB')

#     return adjusted_image

