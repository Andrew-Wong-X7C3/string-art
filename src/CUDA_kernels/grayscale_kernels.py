import math
from numba import cuda, float32
from numba.core.errors import NumbaPerformanceWarning

import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


# ====================================================================================
# function to init global variables
# ====================================================================================
def init_globals(num_points):
    global NUM_POINTS
    NUM_POINTS = num_points


# ====================================================================================
# function to reset search map values to 0
# ====================================================================================
@cuda.jit
def reset_search_map(search_map):
    
    '''
    Assign work as follows:
        - thread.x = pin 2
        NOTE: Assume points < 1024
    '''
    block_idx = cuda.blockIdx.x 
    thread_idx = cuda.threadIdx.x

    # reset to 0
    if block_idx == 0:
        if thread_idx < NUM_POINTS:
            search_map[thread_idx] = 0


# ====================================================================================
# function to calculate benefit from all possible connections
# ====================================================================================
@cuda.jit
def greedy_search(search_map, rr_map, cc_map, vl_map, length_map, normalizer_map,
                edge_encoding, connection_encoding, p1):

    '''
    Assign work as follows:
        - block.x = pin 2 index
        - thread.x = worker to sum line
    '''
    block_idx = cuda.blockIdx.x
    thread_idx = cuda.threadIdx.x

    # loop-stride pins
    p2 = block_idx
    while p2 < NUM_POINTS:

        # get intersecting pixels
        p1_val = int(p1[0])
        length = length_map[p1_val, p2]
        normalizer = normalizer_map[p1_val, p2]

        rr = rr_map[p1_val, p2][:length]
        cc = cc_map[p1_val, p2][:length]
        vl = vl_map[p1_val, p2][:length]

        # check for invalid connection
        if (connection_encoding[p1_val, p2] == 1) or (p1_val == p2):
            search_map[p2] = -math.inf
            return

        # loop-stride length of pixel coordinates
        ix = thread_idx
        while ix < length:
            
            # determine benefit or penalty for drawing over pixel
            pixel = edge_encoding[rr[ix], cc[ix]]
            value = vl[ix]

            benefit = min(pixel, value)
            penalty = min(pixel - value, 0)
            net = (benefit + penalty) / normalizer

            cuda.atomic.add(search_map, p2, net)

            ix += cuda.blockDim.x
        p2 += cuda.gridDim.x


# ====================================================================================
# function to calculate argmax to determine optimal line parameters
# ====================================================================================
@cuda.jit
def argmax_reduction(search_map, benefit, p2, counter):

    '''
    Assign work as follows:
        - thread.x = pin 2 index
    '''
    block_idx = cuda.blockIdx.x
    thread_idx = cuda.threadIdx.x

    # perform atomic max and sync
    counter_val = int(counter[0])
    if block_idx == 0:
        if thread_idx < NUM_POINTS:
            cuda.atomic.max(benefit, counter_val, search_map[thread_idx])
    cuda.syncthreads()

    # store argmax results
    if block_idx == 0:
        if thread_idx < NUM_POINTS:
            if search_map[thread_idx] == benefit[counter_val]:
                p2[0] = thread_idx


# ====================================================================================
# function to draw line onto canvas
# ====================================================================================
@cuda.jit
def draw_line(rr_map, cc_map, vl_map, length_map, edge_encoding, edge_canvas, p1, p2):
    
    '''
    Assign work as follows:
        - idx = index in line spanning from pin 1 to pin 2
    '''
    block_idx = cuda.blockIdx.x
    block_dim = cuda.blockDim.x
    thread_idx = cuda.threadIdx.x
    idx = thread_idx + block_idx * block_dim

    # index arrays
    p1_val = int(p1[0])
    p2_val = int(p2[0])

    length = length_map[p1_val, p2_val]
    rr = rr_map[p1_val, p2_val][:length]
    cc = cc_map[p1_val, p2_val][:length]
    vl = vl_map[p1_val, p2_val][:length]

    # update image assuming length less than 1024 * 1024
    if idx < length:
        pixel = max(edge_encoding[rr[idx], cc[idx]] - vl[idx], 0)
        edge_encoding[rr[idx], cc[idx]] = pixel
        edge_canvas[rr[idx], cc[idx]] = min(vl[idx] + edge_canvas[rr[idx], cc[idx]], 1)


# ====================================================================================
# function to update search variables with new line
# ====================================================================================
@cuda.jit
def update_arrays(path, connection_encoding, p1, p2, counter):

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
            p1_val = int(p1[0])
            p2_val = int(p2[0])
            counter_val = int(counter[0])

            # update path and encoding map
            path[counter_val] = (p1_val, p2_val)
            connection_encoding[p1_val, p2_val] = 1

            # update pins and increment counter
            p1[0] = p2_val
            counter[0] += 1