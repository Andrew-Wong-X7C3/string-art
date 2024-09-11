import numpy as np
from numba import cuda, float32
from numba.core.errors import NumbaPerformanceWarning
import warnings
from tqdm import tqdm

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

# ====================================================================================
# function to reset search map values to 0
# ====================================================================================
@cuda.jit
def reset_search_map(search_map):

    '''
    Assign work as follows:
        - block.x = pin 2 index to reset
        - thread.x = color index
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
def greedy_search(search_map, rr_map, cc_map, length_map, img_encoding, canvas_encoding, connection_encoding, color_weights, temporal_weights, p1):

    '''
    Assign work as follows:
        - block.y = pin 2 index
        - block.x = color index
        - thread.x = worker to sum line
        NOTE: cannot convert encoding data to shared array due to size
    '''
    block_idy = cuda.blockIdx.y
    block_idx = cuda.blockIdx.x
    thread_idx = cuda.threadIdx.x

    # pre-load shared data
    s_color_weights = cuda.shared.array((NUM_COLORS), dtype=float32)
    s_temporal_weights = cuda.shared.array((NUM_COLORS), dtype=float32)

    if block_idx == 0:
        if thread_idx < NUM_COLORS:
            s_color_weights[thread_idx] = color_weights[thread_idx]
            s_temporal_weights[thread_idx] = temporal_weights[thread_idx]

    # loop-stride pins
    p2 = block_idy
    while p2 < NUM_POINTS:

        # loop-stride colors
        ic = block_idx
        while ic < NUM_COLORS:

            # get intersecting pixels
            p1_val = int(p1[ic])
            length = length_map[p1_val, p2]
            rr = rr_map[p1_val, p2][:length]
            cc = cc_map[p1_val, p2][:length]

            # check for invalid connection
            if connection_encoding[p1_val, p2, ic] == 1:
                return
            if p1_val == p2:
                return

            # loop-stride length of pixel coordinates
            ix = thread_idx
            while ix < length:
                
                # check if pixel should be updated by line
                img_pixel = img_encoding[rr[ix], cc[ix]]
                canvas_pixel = canvas_encoding[rr[ix], cc[ix]]

                if (img_pixel == ic) and (canvas_pixel != ic):
                    benefit = (s_color_weights[ic] * s_temporal_weights[ic]) / length
                    cuda.atomic.add(search_map, (p2, ic), benefit)
                
                ix += cuda.blockDim.x
            ic += cuda.gridDim.x
        p2 += cuda.gridDim.y


# ====================================================================================
# function to calculate argmax to determine optimal line parameters
# ====================================================================================
@cuda.jit
def argmax_reduction(search_map, p2, ic):

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
    if block_idx < NUM_COLORS:
        if thread_idx < NUM_POINTS:
            cuda.atomic.max(p2, 0, search_map[thread_idx, block_idx])
    cuda.syncthreads()

    # store argmax results
    if block_idx < NUM_COLORS:
        if thread_idx < NUM_POINTS:
            if search_map[thread_idx, block_idx] == p2[0]:
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

    # update image assuming length less than 1024 * 1024
    if idx < length:
        canvas_encoding[rr[idx], cc[idx]] = ic_val


# ====================================================================================
# function to update search variables with new line
# ====================================================================================
@cuda.jit
def update_arrays(temporal_weights, path, connection_encoding, p1, p2, ic, counter):

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

            # update pins and counter
            p1[ic_val] = p2[0]
            p2[0] = 0
            counter[0] += 1

    # update temporal weights
    if thread_idx < NUM_COLORS:
        pass
        temporal_weights[thread_idx] = 1
    cuda.syncthreads()

    if thread_idx == 0:
        temporal_weights[ic_val] -= 0.01
        if temporal_weights[ic_val] < 1:
            temporal_weights[ic_val] = 1

# =================================================================================================================================
# =================================================================================================================================

class SearchManager:
    def __init__(self):
        self.string_img = None

        self.rr_map_cu = None
        self.cc_map_cu = None
        self.vl_map_cu = None
        self.length_map_cu = None

        self.img_encoding_cu = None
        self.canvas_encoding_cu = None

        self.color_weights_cu = None
        self.temporal_weights_cu = None

        self.path_cu = None
        self.search_map_cu = None
        self.connection_encoding_cu = None

        self.counter_cu = None
        self.p1_cu = None
        self.p2_cu = None
        self.ic_cu = None

    # ====================================================================================
    # function to initialize global variables
    # ====================================================================================
    def init_globals(self, num_colors, num_points, num_lines):
        global NUM_COLORS
        global NUM_POINTS
        global NUM_LINES
        NUM_COLORS = num_colors
        NUM_POINTS = num_points
        NUM_LINES = num_lines

    # ====================================================================================
    # function to load arrays onto GPU shared memory
    # ====================================================================================
    def create_GPU_objects(self, img_dim, rr_map, cc_map, vl_map, length_map, img_encoding, color_weights):

        print('Creating GPU Arrays...')
        self.rr_map_cu = cuda.to_device(rr_map)
        self.cc_map_cu = cuda.to_device(cc_map)
        self.vl_map_cu = cuda.to_device(vl_map)
        self.length_map_cu = cuda.to_device(length_map)

        self.img_encoding_cu = cuda.to_device(img_encoding)
        self.canvas_encoding_cu = cuda.to_device(-np.ones((img_dim, img_dim)))

        self.color_weights_cu = cuda.to_device(color_weights)
        self.temporal_weights_cu = cuda.to_device(color_weights)

        self.path_cu = cuda.to_device(-np.ones((NUM_LINES, 3)))
        self.search_map_cu = cuda.to_device(np.zeros((NUM_POINTS, NUM_COLORS)))
        self.connection_encoding_cu = cuda.to_device(np.zeros((NUM_POINTS, NUM_POINTS, NUM_COLORS)))

        self.counter_cu = cuda.to_device(np.zeros(1))
        self.p1_cu = cuda.to_device(np.zeros(NUM_COLORS))
        self.p2_cu = cuda.to_device(np.zeros(1))
        self.ic_cu = cuda.to_device(np.zeros(1))
        print('Done')

    
    # ====================================================================================
    # helper functions to invoke kernel
    # ====================================================================================
    def reset_search_map(self):
        reset_search_map[1024, 1024](self.search_map_cu)
    def greedy_search(self):
        greedy_search[(128, 8), 1024](self.search_map_cu, self.rr_map_cu, self.cc_map_cu, self.length_map_cu, self.img_encoding_cu, self.canvas_encoding_cu, self.connection_encoding_cu, self.color_weights_cu, self.temporal_weights_cu, self.p1_cu)
    def argmax_reduction(self):
        argmax_reduction[1024, 1024](self.search_map_cu, self.p2_cu, self.ic_cu)
    def draw_line(self):
        draw_line[1024, 1024](self.rr_map_cu, self.cc_map_cu, self.vl_map_cu, self.length_map_cu, self.canvas_encoding_cu, self.p1_cu, self.p2_cu, self.ic_cu)
    def update_arrays(self):
        update_arrays[1, 1024](self.temporal_weights_cu, self.path_cu, self.connection_encoding_cu, self.p1_cu, self.p2_cu, self.ic_cu, self.counter_cu)

    # ====================================================================================
    # function to search
    # ====================================================================================
    def iterate(self):
        print('Drawing Lines...')
        for i in tqdm(range(NUM_LINES)):
            self.reset_search_map()
            self.greedy_search()
            self.argmax_reduction()
            self.draw_line()
            self.update_arrays()
        print('Done')

    # ====================================================================================
    # function to create final image
    # ====================================================================================
    def compose_img(self, palette):
        print('Composing Image...')
        canvas_encoding = self.canvas_encoding_cu.copy_to_host()
        string_img = np.zeros((canvas_encoding.shape[0], canvas_encoding.shape[1], 3))

        for i in range(NUM_COLORS):
            mask = (canvas_encoding == i)
            string_img[mask] = palette[i]

        self.string_img = string_img
        print('Done...')