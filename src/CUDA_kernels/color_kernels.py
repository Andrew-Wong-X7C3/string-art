import math
from numba import cuda, float32
from numba.core.errors import NumbaPerformanceWarning

import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


# ====================================================================================
# function to init global variables
# ====================================================================================
def init_globals(num_colors, num_points, num_lines):
    global NUM_COLORS
    global NUM_POINTS
    global NUM_LINES

    NUM_COLORS = num_colors
    NUM_POINTS = num_points
    NUM_LINES = num_lines


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
                color_encoding, color_canvas, connection_encoding,
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
                img_pixel = int(color_encoding[rr[ix], cc[ix]])
                canvas_pixel = int(color_canvas[rr[ix], cc[ix]])

                # incorrect canvas color, should be drawn over
                if (img_pixel != canvas_pixel) and (img_pixel == ic):
                    benefit = vl[ix] * (s_color_weights[ic] * s_temporal_weights[ic]) / normalizer
                    cuda.atomic.add(search_map, (p2, ic), benefit)

                # correct canvas color, should not be drawn over
                if (img_pixel == canvas_pixel) and (img_pixel != ic):
                    penalty = -vl[ix] * (s_color_weights[img_pixel] / s_color_weights[ic]) / normalizer
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
def draw_line(rr_map, cc_map, vl_map, length_map, color_canvas, p1, p2, ic):
    
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
        color_canvas[rr[idx], cc[idx]] = ic_val


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
            2 * math.pow(
                2 - 
                math.pow(
                    counter_val / NUM_LINES,
                    1 / (math.log(color_weights[thread_idx] + 1))
                ),
                math.log(color_weights[thread_idx] + 1)
            )
        
    cuda.syncthreads()
    if thread_idx == 0:
        pass

# ========================================================================================================================
# ========================================================================================================================