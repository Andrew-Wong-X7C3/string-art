import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from numba import cuda

from src.CUDA_kernels import color_kernels
from src.CUDA_kernels import grayscale_kernels


class SearchManager:
    def __init__(self):
        self.string_img = None

        self.rr_map = None
        self.cc_map = None
        self.vl_map = None
        self.length_map = None
        self.normalizer_map = None

        self.edge_encoding = None
        self.color_encoding = None

        self.edge_canvas = None
        self.color_canvas = None

        self.edge_path = None
        self.color_path = None

        self.color_weights = None
        self.temporal_weights = None

        self.search_map = None
        self.connection_encoding = None
        self.counter = None
        self.p1 = None
        self.p2 = None
        self.ic = None

    # ====================================================================================
    # function to initialize global variables
    # ====================================================================================
    def init_globals(self, num_colors, num_points, num_edge_lines, num_color_lines, max_dimension):

        global NUM_COLORS
        global NUM_POINTS
        global NUM_EDGE_LINES
        global NUM_COLOR_LINES
        global MAX_STRING_DIMENSION

        NUM_COLORS = num_colors
        NUM_POINTS = num_points
        NUM_EDGE_LINES = num_edge_lines
        NUM_COLOR_LINES = num_color_lines
        MAX_STRING_DIMENSION = max_dimension

        grayscale_kernels.init_globals(NUM_POINTS)
        color_kernels.init_globals(NUM_COLORS, NUM_POINTS, NUM_COLOR_LINES)
    # ====================================================================================
    # function to load arrays onto GPU shared memory
    # ====================================================================================
    def create_GPU_objects(self, rr_map, cc_map, vl_map, length_map, normalizer_map, edge_values, img_encoding, color_weights):

        print('Creating GPU Arrays...')
        self.rr_map = cuda.to_device(rr_map)
        self.cc_map = cuda.to_device(cc_map)
        self.vl_map = cuda.to_device(vl_map)
        self.length_map = cuda.to_device(length_map)
        self.normalizer_map = cuda.to_device(normalizer_map)

        self.edge_values = cuda.to_device(edge_values)
        self.color_encoding = cuda.to_device(img_encoding)
        self.color_weights = cuda.to_device(color_weights)
        self.temporal_weights = cuda.to_device(color_weights)
        print('Done')


    # ====================================================================================
    # function to search color lines
    # ====================================================================================
    def draw_color(self):

        print('Drawing Color Lines...')

        #create GPU objects for color search
        self.color_canvas = cuda.to_device(-np.ones((MAX_STRING_DIMENSION, MAX_STRING_DIMENSION)))
        self.color_path = cuda.to_device(-np.ones((NUM_COLOR_LINES, 3)))
        self.search_map = cuda.to_device(np.zeros((NUM_POINTS, NUM_COLORS)))
        self.connection_encoding = cuda.to_device(np.zeros((NUM_POINTS, NUM_POINTS, NUM_COLORS)))
        self.counter = cuda.to_device(np.zeros(1))
        self.benefit = cuda.to_device(-np.inf * np.ones(NUM_COLOR_LINES))
        self.p1 = cuda.to_device(np.zeros(NUM_COLORS))
        self.p2 = cuda.to_device(np.zeros(1))
        self.ic = cuda.to_device(np.zeros(1))

        # iterate
        for i in tqdm(range(NUM_COLOR_LINES)):
            color_kernels.reset_search_map[1024, 1024](self.search_map)
            color_kernels.greedy_search[(8, 128), 1024](self.search_map, self.rr_map, self.cc_map, self.vl_map, self.length_map, self.normalizer_map,
                                      self.color_encoding, self.color_canvas, self.connection_encoding,
                                      self.color_weights, self.temporal_weights,
                                      self.p1)
            color_kernels.argmax_reduction[1024, 1024](self.search_map, self.benefit, self.p2, self.ic, self.counter)
            color_kernels.draw_line[1024, 1024](self.rr_map, self.cc_map, self.vl_map, self.length_map, self.color_canvas, self.p1, self.p2, self.ic)
            color_kernels.update_arrays[1, 1024](self.temporal_weights, self.color_weights, self.color_path, self.connection_encoding, self.p1, self.p2, self.ic, self.counter)  
        print('Done')

    # ====================================================================================
    # function to search grayscale edges
    # ====================================================================================
    def draw_edge(self):

        print('Drawing Edge Lines...')

        #create GPU objects for edge search
        self.edge_canvas = cuda.to_device(np.zeros((MAX_STRING_DIMENSION, MAX_STRING_DIMENSION)))
        self.edge_path = cuda.to_device(-np.ones((NUM_EDGE_LINES, 2)))
        self.search_map = cuda.to_device(np.zeros((NUM_POINTS)))
        self.connection_encoding = cuda.to_device(np.zeros((NUM_POINTS, NUM_POINTS)))
        self.counter = cuda.to_device(np.zeros(1))
        self.benefit = cuda.to_device(-np.inf * np.ones(NUM_EDGE_LINES))
        self.p1 = cuda.to_device(np.zeros(1))
        self.p2 = cuda.to_device(np.zeros(1))

        # iterate
        for i in tqdm(range(NUM_EDGE_LINES)):
            grayscale_kernels.reset_search_map[1, 1024](self.search_map)
            grayscale_kernels.greedy_search[1024, 1024](self.search_map, self.rr_map, self.cc_map, self.vl_map, self.length_map, self.normalizer_map, self.edge_values, self.connection_encoding, self.p1)
            grayscale_kernels.argmax_reduction[1, 1024](self.search_map, self.benefit, self.p2, self.counter)
            grayscale_kernels.draw_line[1024, 1024](self.rr_map, self.cc_map, self.vl_map, self.length_map, self.edge_values, self.edge_canvas, self.p1, self.p2)
            grayscale_kernels.update_arrays[10, 1024](self.edge_path, self.connection_encoding, self.p1, self.p2, self.counter)
        print('Done')



    # ====================================================================================
    # function to create final image
    # ====================================================================================
    def compose_img(self, palette, output_folder):

        print('Composing Image...')
        edge_canvas = self.edge_canvas.copy_to_host()
        color_canvas = self.color_canvas.copy_to_host()
        string_img = np.zeros((color_canvas.shape[0], color_canvas.shape[1], 3))

        # add color
        for i in range(NUM_COLORS):
            mask = (color_canvas == i)
            string_img[mask] = palette[i]

        # add edges
        string_img[edge_canvas.astype(bool)] = [1, 1, 1]

        # save results
        self.string_img = string_img
        plt.imsave(os.path.join(output_folder, 'string_{}_{}_{}_{}.jpg'.format(NUM_COLORS, NUM_POINTS, NUM_EDGE_LINES, NUM_COLOR_LINES)), self.string_img)
        np.savetxt(os.path.join(output_folder, 'path.txt'), self.color_path.copy_to_host(), fmt='%d')
        print('Done...')


    # ====================================================================================
    # function to generate contribution chart for each color
    # ====================================================================================
    def generate_chart(self, palette, output_folder):

        print('Generating Chart...')
        print('\tCopying to Host...')
        path = self.color_path.copy_to_host()
        color_canvas = self.color_canvas.copy_to_host()
        rr_map = self.rr_map.copy_to_host()
        cc_map = self.cc_map.copy_to_host()
        vl_map = self.vl_map.copy_to_host()
        length_map = self.length_map.copy_to_host()

        print('\tIterating Colors...')
        fig, ax = plt.subplots(3, NUM_COLORS, figsize=(30, 10))
        for i in tqdm(range(NUM_COLORS)):
            partial_string_img = np.zeros((MAX_STRING_DIMENSION, MAX_STRING_DIMENSION))
            for j in range(path.shape[0]):
                p1, p2, ic = path[j].astype(int)
                if ic == i:
                    length = length_map[p1, p2]
                    rr = rr_map[p1, p2][:length]
                    cc = cc_map[p1, p2][:length]
                    partial_string_img[rr, cc] = 1

            ax[0][i].imshow(partial_string_img[::2, ::2])
            ax[1][i].imshow((color_canvas == i)[::2, ::2], cmap='binary_r')
            ax[2][i].imshow([[palette[i]]])

            ax[0][i].axis('off')
            ax[1][i].axis('off')
            ax[2][i].axis('off')
        
        plt.savefig(os.path.join(output_folder, 'chart.jpg'), bbox_inches='tight', dpi=300)
        plt.close(fig)
        print('Done')