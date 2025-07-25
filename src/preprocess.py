import os
from tqdm import tqdm
from PIL import Image

import numpy as np
import scipy as sp
from numba import jit
from numba_progress import ProgressBar

import faiss
from sklearn.neighbors import KDTree
from fast_colorthief import get_palette


class PreProcessManager:
    def __init__(self):
        self.img = None
        self.edge = None
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
    # function to identify edges using 2D Gaussian derivative
    # ====================================================================================
    def gaussian_edge(self, sigma):

        # copy variables
        print('Applying Gaussian Edge Detection...')
        img = self.img.copy()

        # helper function to calculate 2D Gaussian derivative
        def gaussDeriv2D(sigma):
            Gx = np.zeros((6 * sigma + 1, 6 * sigma + 1))
            mask_range = np.arange(int(-np.ceil(3 * sigma)), int(np.ceil(3 * sigma)) + 1)
            for iy, ix in np.ndindex(Gx.shape):
                x = mask_range[ix]
                y = mask_range[iy]
                c = 1 / (2 * np.pi * np.power(sigma, 4))
                exp = -(np.power(x, 2) + np.power(y, 2)) / (2 * np.power(sigma, 2))
                Gx[iy, ix] = -x * c * np.exp(exp)
            return Gx, Gx.T

        # helper function to convert to gray scale
        def rgb2gray(rgb):
            return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


        # apply filter per RGB channel
        Gx, Gy = gaussDeriv2D(sigma)
        edge_img = np.zeros(img.shape)
        for channel in tqdm(range(img.shape[-1])):
            layer = img[:, :, channel]
            gx_img = sp.ndimage.convolve(layer, Gx, mode='nearest')
            gy_img = sp.ndimage.convolve(layer, Gy, mode='nearest')
            edge_img[:, :, channel] = np.sqrt(np.power(gx_img, 2) + np.power(gy_img, 2))

        # normalize, convert to grayscale, and clip to 1 std
        edge_img /= np.max(edge_img)
        edge_img = rgb2gray(edge_img)
        edge_img[edge_img < np.mean(edge_img) + np.std(edge_img)] = 0

        # return results
        self.edge = edge_img
        print('Done')


    # ====================================================================================
    # function to circle crop
    # ====================================================================================
    def circle_crop(self, coverage, offset_x, offset_y):

        # calculate radius
        print('Cropping Image...')
        img = self.img.copy()
        edge = self.edge.copy()
        radius = int(np.min(img.shape[:2]) * coverage / 2)

        # create binary 2D mask
        x_tile = np.tile(range(img.shape[1]), (img.shape[0], 1))
        y_tile = np.tile(range(img.shape[0]), (img.shape[1], 1)).T

        x_tile -= int(img.shape[1] / 2) + offset_x
        y_tile -= int(img.shape[0] / 2) + offset_y

        mask = np.sqrt(np.power(x_tile, 2) + np.power(y_tile, 2)) < radius

        # apply mask to image and crop
        x_edge = int((img.shape[1] - 2 * radius) / 2)
        y_edge = int((img.shape[0] - 2 * radius) / 2)
        
        cropped_img = np.where(mask[..., None], img, 1)
        cropped_img = cropped_img[y_edge + offset_y:-y_edge + offset_y, x_edge + offset_x:-x_edge + offset_x, :]

        cropped_edge = np.where(mask, edge, 1)
        cropped_edge = cropped_edge[y_edge + offset_y:-y_edge + offset_y, x_edge + offset_x:-x_edge + offset_x]

        # account for rounding error and force equal dimensions
        dim = np.min(cropped_img.shape[:-1])
        cropped_img = cropped_img[:dim, :dim]
        cropped_edge = cropped_edge[:dim, :dim]

        # return radius and image
        self.radius = radius
        self.img = cropped_img
        self.edge = cropped_edge
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
        print('Resizing to Max Dimension...')
        img = self.img.copy()
        edge = self.edge.copy()
        radius = self.radius
        palette = self.palette

        def resize(x):
            resized = Image.fromarray(np.uint8(x * 255))
            ratio = max_dimension / np.max(resized.size)
            resized_radius = int(radius * ratio)
            dim = (np.array(resized.size) * ratio).astype(int)
            resized = resized.resize(dim, Image.NEAREST)
            resized = np.array(resized) / 255
            return resized, resized_radius

        # upscale edges
        print('\tResizing Edges and Canvas...')
        resized_edge, _ = resize(edge)
        resized_img, resized_radius = resize(img)

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
        self.img = resized_img
        self.edge = resized_edge
        self.radius = resized_radius
        print('Done')

