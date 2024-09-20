import os
import math
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from numba import cuda, float32
from numba.core.errors import NumbaPerformanceWarning

import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

# ====================================================================================
# function to reset search map values to 0
# ====================================================================================
@cuda.jit
def reset_search_map(search_map):

    '''
    Assign work as follows:
        - block.x = pin 2 index
        - thread.x = color index
        NOTE: assume colors < 1024 and points < 1024
    '''
    block_idx = cuda.blockIdx.x 
    thread_idx = cuda.threadIdx.x

    # reset to 0
    if block_idx < NUM_POINTS:
        if thread_idx < NUM_COLORS:
            search_map[block_idx, thread_idx] = 0


# ====================================================================================
# function to calculate benefit from all possible connections
# ====================================================================================
@cuda.jit
def greedy_search(search_map, rr_map, cc_map, vl_map, length_map, normalizer_map,
                  img_encoding, canvas_encoding, connection_encoding,
                  color_weights, temporal_weights,
                  p1):

    '''
    Assign work as follows:
        - block.y = pin 2 index
        - block.x = color index
        - thread.x = worker to sum line
        NOTE: cannot convert encoding data to shared array due to size
        NOTE: assume colors < 1024
    '''
    block_idy = cuda.blockIdx.y
    block_idx = cuda.blockIdx.x
    thread_idx = cuda.threadIdx.x

    # pre-load shared data for each block
    s_color_weights = cuda.shared.array((NUM_COLORS), dtype=float32)
    s_temporal_weights = cuda.shared.array((NUM_COLORS), dtype=float32)

    if thread_idx < NUM_COLORS:
        s_color_weights[thread_idx] = color_weights[thread_idx]
        s_temporal_weights[thread_idx] = temporal_weights[thread_idx]
    cuda.syncthreads()

    # loop-stride pins
    p2 = block_idy
    while p2 < NUM_POINTS:

        # loop-stride colors
        ic = block_idx
        while ic < NUM_COLORS:

            # get intersecting pixels
            p1_val = int(p1[ic])
            length = length_map[p1_val, p2]
            normalizer = normalizer_map[p1_val, p2]

            rr = rr_map[p1_val, p2][:length]
            cc = cc_map[p1_val, p2][:length]
            vl = vl_map[p1_val, p2][:length]

            # check for invalid connection
            if (connection_encoding[p1_val, p2, ic] == 1) or (p1_val == p2):
                search_map[p2, ic] = -math.inf
                return

            # loop-stride length of pixel coordinates
            ix = thread_idx
            while ix < length:
                
                # get corresponding encoding pixels
                img_pixel = int(img_encoding[rr[ix], cc[ix]])
                canvas_pixel = int(canvas_encoding[rr[ix], cc[ix]])

                # incorrect canvas color, should be drawn over
                if (img_pixel != canvas_pixel) and (img_pixel == ic):
                    benefit = vl[ix] * (s_color_weights[ic] * s_temporal_weights[ic]) / normalizer
                    cuda.atomic.add(search_map, (p2, ic), benefit)

                # correct canvas color, should not be drawn over
                if (img_pixel == canvas_pixel) and (img_pixel != ic):
                    # penalty = -vl[ix] * (1 / s_color_weights[ic]) / normalizer
                    penalty = -vl[ix] * s_color_weights[img_pixel] / normalizer
                    cuda.atomic.add(search_map, (p2, ic), penalty)

                ix += cuda.blockDim.x
            ic += cuda.gridDim.x
        p2 += cuda.gridDim.y


# ====================================================================================
# function to calculate argmax to determine optimal line parameters
# ====================================================================================
@cuda.jit
def argmax_reduction(search_map, benefit, p2, ic, counter):

    '''
    Assign work as follows:
        - block.x = pin 2 index
        - thread.x = color index
        NOTE: possible race condition due to no grid sync
        NOTE: assignments switched due to race condition
    '''
    block_idx = cuda.blockIdx.x
    thread_idx = cuda.threadIdx.x

    # perform atomic max and sync
    counter_val = int(counter[0])
    if block_idx < NUM_COLORS:
        if thread_idx < NUM_POINTS:
            cuda.atomic.max(benefit, counter_val, search_map[thread_idx, block_idx])
    cuda.syncthreads()

    # store argmax results
    if block_idx < NUM_COLORS:
        if thread_idx < NUM_POINTS:
            if search_map[thread_idx, block_idx] == benefit[counter_val]:
                p2[0] = thread_idx
                ic[0] = block_idx


# ====================================================================================
# function to draw line onto canvas
# ====================================================================================
@cuda.jit
def draw_line(rr_map, cc_map, vl_map, length_map, canvas_encoding, p1, p2, ic):
    
    '''
    Assign work as follows:
        - idx = index in line spanning from pin 1 to pin 2
        NOTE: assume length < 1024 * 1024
    '''
    block_idx = cuda.blockIdx.x
    block_dim = cuda.blockDim.x
    thread_idx = cuda.threadIdx.x
    idx = thread_idx + block_idx * block_dim

    # index arrays
    ic_val = int(ic[0])
    p1_val = int(p1[ic_val])
    p2_val = int(p2[0])

    length = length_map[p1_val, p2_val]
    rr = rr_map[p1_val, p2_val][:length]
    cc = cc_map[p1_val, p2_val][:length]
    vl = vl_map[p1_val, p2_val][:length]

    # update image
    if idx < length:
        canvas_encoding[rr[idx], cc[idx]] = ic_val


# ====================================================================================
# function to update search variables with new line
# ====================================================================================
@cuda.jit(fastmath=True)
def update_arrays(temporal_weights, color_weights, path, connection_encoding, p1, p2, ic, counter):

    '''
    Assign work as follows:
        - block 0, thread 0: serial update path, map, and pins
        - thread.x: increment temporal weights
    '''
    block_idx = cuda.blockIdx.x
    thread_idx = cuda.threadIdx.x

    # single thread operation
    if block_idx == 0:
        if thread_idx == 0:

            # index arrays
            ic_val = int(ic[0])
            p2_val = int(p2[0])
            p1_val = int(p1[ic_val])
            counter_val = int(counter[0])

            # update path and encoding map
            path[counter_val] = (p1_val, p2_val, ic_val)
            connection_encoding[p1_val, p2_val, ic_val] = 1

            # update pins and increment counter
            p1[ic_val] = p2_val
            counter[0] += 1

    # update temporal weights
    if thread_idx < NUM_COLORS:
        temporal_weights[thread_idx] = \
            math.pow(
                2 - 
                math.pow(
                    counter_val / NUM_LINES,
                    1 / (counter_val * math.log(color_weights[thread_idx] + 1))
                ),
                math.log(color_weights[thread_idx] + 1)
            )
        
    cuda.syncthreads()
    if thread_idx == 0:
        pass


# =================================================================================================================================
# =================================================================================================================================

class SearchManager:
    def __init__(self):

        self.i = None
        self.string_img = None

        self.rr_map = None
        self.cc_map = None
        self.vl_map = None
        self.length_map = None
        self.normalizer_map = None

        self.img_encoding = None
        self.canvas_encoding = None

        self.color_weights = None
        self.temporal_weights = None

        self.path = None
        self.search_map = None
        self.connection_encoding = None

        self.counter = None
        self.p1 = None
        self.p2 = None
        self.ic = None

    # ====================================================================================
    # function to initialize global variables
    # ====================================================================================
    def init_globals(self, num_colors, num_points, num_lines, max_dimension):

        global NUM_COLORS
        global NUM_POINTS
        global NUM_LINES
        global MAX_STRING_DIMENSION
        NUM_COLORS = num_colors
        NUM_POINTS = num_points
        NUM_LINES = num_lines
        MAX_STRING_DIMENSION = max_dimension

    # ====================================================================================
    # function to load arrays onto GPU shared memory
    # ====================================================================================
    def create_GPU_objects(self, img_dim, rr_map, cc_map, vl_map, length_map, normalizer_map, img_encoding, color_weights):

        print('Creating GPU Arrays...')
        self.rr_map = cuda.to_device(rr_map)
        self.cc_map = cuda.to_device(cc_map)
        self.vl_map = cuda.to_device(vl_map)
        self.length_map = cuda.to_device(length_map)
        self.normalizer_map = cuda.to_device(normalizer_map)

        self.img_encoding = cuda.to_device(img_encoding)
        self.canvas_encoding = cuda.to_device(-np.ones((img_dim, img_dim)))

        self.color_weights = cuda.to_device(color_weights)
        self.temporal_weights = cuda.to_device(color_weights)

        self.path = cuda.to_device(-np.ones((NUM_LINES, 3)))
        self.search_map = cuda.to_device(np.zeros((NUM_POINTS, NUM_COLORS)))
        self.connection_encoding = cuda.to_device(np.zeros((NUM_POINTS, NUM_POINTS, NUM_COLORS)))

        self.counter = cuda.to_device(np.zeros(1))
        self.benefit = cuda.to_device(-np.inf * np.ones(NUM_LINES))
        self.p1 = cuda.to_device(np.zeros(NUM_COLORS))
        self.p2 = cuda.to_device(np.zeros(1))
        self.ic = cuda.to_device(np.zeros(1))
        print('Done')

    
    # ====================================================================================
    # helper functions to invoke kernel
    # ====================================================================================
    def reset_search_map(self):
        reset_search_map[1024, 1024](self.search_map)
    def greedy_search(self):
        greedy_search[(8, 128), 1024](self.search_map, self.rr_map, self.cc_map, self.vl_map, self.length_map, self.normalizer_map,
                                      self.img_encoding, self.canvas_encoding, self.connection_encoding,
                                      self.color_weights, self.temporal_weights,
                                      self.p1)
    def argmax_reduction(self):
        argmax_reduction[1024, 1024](self.search_map, self.benefit, self.p2, self.ic, self.counter)
    def draw_line(self):
        draw_line[1024, 1024](self.rr_map, self.cc_map, self.vl_map, self.length_map, self.canvas_encoding, self.p1, self.p2, self.ic)
    def update_arrays(self):
        update_arrays[1, 1024](self.temporal_weights, self.color_weights, self.path, self.connection_encoding, self.p1, self.p2, self.ic, self.counter)


    # ====================================================================================
    # function to search
    # ====================================================================================
    def iterate(self):

        temp = np.zeros((NUM_LINES, NUM_COLORS))

        print('Drawing Lines...')
        for self.i in tqdm(range(NUM_LINES)):
            self.reset_search_map()
            self.greedy_search()

            x = self.search_map.copy_to_host()
            temp[self.i] = np.max(x, axis=0)

            self.argmax_reduction()
            self.draw_line()
            self.update_arrays()
        print('Done')

        np.savetxt('temp.txt', temp, fmt='%f')


    # ====================================================================================
    # function to create final image
    # ====================================================================================
    def compose_img(self, palette, output_folder):

        print('Composing Image...')
        canvas_encoding = self.canvas_encoding.copy_to_host()
        string_img = np.zeros((canvas_encoding.shape[0], canvas_encoding.shape[1], 3))

        for i in range(NUM_COLORS):
            mask = (canvas_encoding == i)
            string_img[mask] = palette[i]

        self.string_img = string_img
        plt.imsave(os.path.join(output_folder, 'string_{}_{}_{}.jpg'.format(NUM_COLORS, NUM_POINTS, NUM_LINES)), self.string_img)
        np.savetxt(os.path.join(output_folder, 'path.txt'), self.path.copy_to_host(), fmt='%d')
        np.savetxt(os.path.join(output_folder, 'benefit.txt'), self.benefit.copy_to_host(), fmt='%f')
        print('Done...')


    # ====================================================================================
    # function to generate contribution chart for each color
    # ====================================================================================
    def generate_chart(self, palette, output_folder):

        print('Generating Chart...')
        print('\tCopying to Host...')
        path = self.path.copy_to_host()
        canvas_encoding = self.canvas_encoding.copy_to_host()
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
            ax[1][i].imshow((canvas_encoding == i)[::2, ::2], cmap='binary_r')
            ax[2][i].imshow([[palette[i]]])

            ax[0][i].axis('off')
            ax[1][i].axis('off')
            ax[2][i].axis('off')
        
        plt.savefig(os.path.join(output_folder, 'chart.jpg'), bbox_inches='tight', dpi=300)
        plt.close(fig)
        print('Done')