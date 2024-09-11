import numpy as np
from skimage.draw import line_aa
from tqdm import tqdm


class QuantizeManager:
    def __init__(self):
        self.rr_map = None
        self.cc_map = None
        self.vl_map = None
        self.length_map = None

        self.color_weights = None
        self.img_encoding = None

    # ====================================================================================
    # function to compute pixel intersection with Bresenham's algorithm
    # ====================================================================================
    def compute_pin_mapping(self, radius, img_dim, num_points):
        
        # helper function to update map
        def update_map(obj_map, length, value, p1, p2):
            obj_map[p1, p2, :length] = value
            obj_map[p2, p1, :length] = value
            return obj_map

        # create 2d array to store int location of pins
        theta = 2 * np.pi / num_points
        points = np.array([(np.cos(theta * i) * radius, np.sin(theta * i) * radius) for i in range(num_points)])
        points += radius
        pins = np.round(points).astype(int).T
        np.putmask(pins, pins == radius * 2, radius * 2 - 1)

        # create data structures to store line data
        max_length = int(3 * img_dim)
        rr_map = np.zeros((num_points, num_points, max_length), dtype=np.int16)
        cc_map = np.zeros((num_points, num_points, max_length), dtype=np.int16)
        vl_map = np.zeros((num_points, num_points, max_length), dtype=np.float16)
        length_map = np.zeros((num_points, num_points), dtype=np.int16)

        # calculate all possible anti-aliased lines using Bresenham's algorithm
        print('Calculating Intersection Using Bresenham\'s algorithm...')
        for p1 in tqdm(range(num_points)):
            for p2 in range(num_points):
                if p2 <= p1:

                    rr, cc, vl = line_aa(pins[0][p1], pins[1][p1], pins[0][p2], pins[1][p2])
                    length = vl.shape[0]
                    length_map[p1, p2] = length
                    length_map[p2, p1] = length

                    rr_map = update_map(rr_map, length, rr, p1, p2)
                    cc_map = update_map(cc_map, length, cc, p1, p2)
                    vl_map = update_map(vl_map, length, vl, p1, p2)

        # return mapping
        self.rr_map = rr_map
        self.cc_map = cc_map
        self.vl_map = vl_map
        self.length_map = length_map
        print('Done')


    # ====================================================================================
    # function to encode image with weight values
    # ====================================================================================
    def encode_img(self, img, palette):

        # assign weights based on pixel frequency
        print('Analyzing Image...')
        print('\tCalculating Weights...')
        total_pixels = 0
        palette_frequency = []

        for i in tqdm(range(palette.shape[0])):
            color = palette[i]
            count = np.sum((img == color).all(axis=2))
            palette_frequency.append([color, count])
            total_pixels += count

        color_weights = np.sqrt([(total_pixels / frequency) for _, frequency in palette_frequency])

        # assign int value to each color in palette
        print('\tEncoding Image Data...')
        img_encoding = np.zeros(img.shape[:-1])
        for i in tqdm(range(len(palette))):
            img_encoding[(img == palette[i]).all(axis=2)] = i

        # return results
        self.color_weights = color_weights
        self.img_encoding = img_encoding
        print('Done')