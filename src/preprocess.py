import os
from tqdm import tqdm
from PIL import Image

import numpy as np
from numba import jit
from numba_progress import ProgressBar

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable

import faiss
from sklearn.neighbors import KDTree
from fast_colorthief import get_palette


class PreProcessManager:
    def __init__(self):
        self.img = None
        self.radius = 0
        self.palette = None


    # ====================================================================================
    # function to load image
    # ====================================================================================
    def load_image(self, path, max_dimension):

        print('Loading Image...')
        img = Image.open(path)
        ratio = max_dimension / np.max(img.size)
        dim = (np.array(img.size) * ratio).astype(int)
        img = img.resize(dim, Image.LANCZOS)
        img = np.array(img) / 255
        self.img = img
        print('Done')


    # ====================================================================================
    # function to circle crop
    # ====================================================================================
    def circle_crop(self, coverage, offset_x, offset_y):

        # calculate radius
        print('Cropping Image...')
        img = self.img.copy()
        radius = int(np.min(img.shape[:2]) * coverage / 2)

        # create binary 2D mask
        x_tile = np.tile(range(img.shape[1]), (img.shape[0], 1))
        y_tile = np.tile(range(img.shape[0]), (img.shape[1], 1)).T

        x_tile -= int(img.shape[1] / 2) + offset_x
        y_tile -= int(img.shape[0] / 2) + offset_y

        mask = np.sqrt(np.power(x_tile, 2) + np.power(y_tile, 2)) < radius

        # apply mask to image and crop
        cropped_img = np.where(mask[..., None], img, 1)
        x_edge = int((img.shape[1] - 2 * radius) / 2)
        y_edge = int((img.shape[0] - 2 * radius) / 2)
        cropped_img = cropped_img[y_edge + offset_y:-y_edge + offset_y, x_edge + offset_x:-x_edge + offset_x, :]

        # account for rounding error and force equal dimensions
        dim = np.min(cropped_img.shape[:-1])
        cropped_img = cropped_img[:dim, :dim]

        # return mask and image
        self.radius = radius
        self.img = cropped_img
        print('Done')


    # ====================================================================================
    # function to reduce color space to n colors
    # ====================================================================================
    def reduce_colors(self, num_colors, output_folder):

        # copy variables
        print('Reducing Colors...')
        img = self.img.copy()
        radius = self.radius

        # use color thief to create palette
        print('\tCalculating Palette...')
        rgba_img = np.dstack((img, np.ones(img.shape[:-1])))
        rgba_img = (rgba_img * 255).astype(np.uint8)
        palette = get_palette(rgba_img, color_count=num_colors, use_gpu=True)
        if len(palette) != num_colors: palette = get_palette(rgba_img, color_count=num_colors + 1, use_gpu=True)
        palette = np.array(palette) / 255

        # identify pixels only in cropped circle
        m_tile = np.tile(range(img.shape[0]), (img.shape[0], 1))
        m_tile -= int(img.shape[0] / 2)
        mask = np.sqrt(np.power(m_tile, 2) + np.power(m_tile.T, 2)) < radius
        object_pixels = img[mask]

        # create GPU indexer
        print('\tCreating GPU Indexer...')
        gpu_resource = faiss.StandardGpuResources()
        index_flat = faiss.IndexFlatL2(len(img.shape))
        gpu_index_flat = faiss.index_cpu_to_gpu(gpu_resource, 0, index_flat)
        gpu_index_flat.add(palette)
        D, I = gpu_index_flat.search(object_pixels, k=1)

        # cluster pixels
        clustered_img = np.ones(img.shape)
        indices = mask.nonzero()
        for i, (iy, ix) in tqdm(enumerate(zip(*indices)), total=len(indices[0])):
            index = I[i]
            clustered_img[iy, ix] = palette[index]

        # return pallette and reduced image
        self.palette = palette
        self.img = clustered_img
        np.savetxt(os.path.join(output_folder, 'palette.txt'), self.palette, fmt='%f')
        print('Done')


    # ====================================================================================
    # function to apply Floyd–Steinberg dithering
    # ====================================================================================
    def dither(self, num_colors):

        # copy variables
        print('Dithering Image...')
        img = self.img.copy()

        # create Floyd-Steinberg error kernel and expand to match channels
        print('\tCalculating Kernel...')
        error_kernel = np.array([
            [0   , 0   , 0   ],
            [0   , 0   , 7/16],
            [3/16, 5/16, 1/16],
        ])
        error_kernel = np.repeat(error_kernel[:, :, np.newaxis], img.shape[-1], axis=2)

        # dither image
        @jit(nopython=True)
        def dither_helper(img, progress_proxy):
            for iy in range(1, img.shape[0] - 1):
                for ix in range(1, img.shape[1] - 1):
                    c_prime = np.round(img[iy, ix] * (num_colors - 1)) / (num_colors - 1)
                    error = img[iy, ix] - c_prime
                    img[iy, ix] = c_prime
                    img[iy - 1 : iy + 2, ix - 1: ix + 2] += error_kernel * error
                    progress_proxy.update(1)
            return np.clip(img, 0, 1)

        dithered_img = img.copy()
        total = (dithered_img.shape[0] - 2) * (dithered_img.shape[1] - 2)

        print('\tAssigning Pixels...')
        with ProgressBar(total=total) as progress:
            dithered_img = dither_helper(dithered_img, progress)

        # return results
        self.img = dithered_img
        print('Done')


    # ====================================================================================
    # function to up-scale image with nearest neighbor interpolation
    # ====================================================================================
    def resize_canvas(self, choices, max_dimension):

        # copy variables
        print('Resizing Canvas...')
        img = self.img.copy()
        radius = self.radius
        palette = self.palette

        # upscale image using nearest neighbor
        resized_img = Image.fromarray(np.uint8(img * 255))
        ratio = max_dimension / np.max(resized_img.size)
        resized_radius = int(radius * ratio)
        dim = (np.array(resized_img.size) * ratio).astype(int)
        resized_img = resized_img.resize(dim, Image.NEAREST)
        resized_img = np.array(resized_img) / 255

        # perform monte carlo simulation to get new up-scaled palette due to rounding error from uint conversion
        print('\tCollecting Monte Carlo Palette...')
        index = (np.random.rand(choices, 2) * (max_dimension - 1)).astype(int).T
        selection = resized_img[index[0], index[1]]
        resized_colors = np.unique(selection.reshape(-1, selection.shape[-1]), axis=0)

        # replace colors with original palette using knn
        print('\tKNN Pixel Assignment...')
        tree = KDTree(resized_colors, leaf_size=2)
        D, I = tree.query(palette, k=1)
        for i in tqdm(range(len(palette))):
            resized_img[(resized_img == resized_colors[I[i][0]]).all(axis=2)] = palette[i]
        
        # return new radius and image
        self.radius = resized_radius
        self.img = resized_img
        print('Done')

