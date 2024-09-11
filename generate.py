import os
from matplotlib import pyplot as plt

from src.preprocess import PreProcessManager
from src.quantize import QuantizeManager
from src.search import SearchManager

os.system('cls||clear')

NUM_COLORS = 8
NUM_POINTS = 180
NUM_LINES = 3000

MAX_LOAD_DIMENSION = 1024
MAX_STRING_DIMENSION = 4096
PATH = 'images/girl_with_pearl_earring.jpg'

BG_FLAG = 0
BG_THRESHOLD = 0
IMG_OFFSET_X = 25
IMG_OFFSET_Y = -50
IMG_COVERAGE = 0.9
MONTE_CARLO_CHOICES = 1000


preprocess = PreProcessManager()
preprocess.load_image(PATH, MAX_LOAD_DIMENSION)
preprocess.circle_crop(IMG_COVERAGE, IMG_OFFSET_X, IMG_OFFSET_Y)
preprocess.reduce_colors(NUM_COLORS)
preprocess.dither(NUM_COLORS)
preprocess.resize_canvas(MONTE_CARLO_CHOICES, MAX_STRING_DIMENSION)

quantize = QuantizeManager()
quantize.compute_pin_mapping(preprocess.radius, MAX_STRING_DIMENSION, NUM_POINTS)
quantize.encode_img(preprocess.img, preprocess.palette)

search = SearchManager()
search.init_globals(NUM_COLORS, NUM_POINTS, NUM_LINES)
search.create_GPU_objects(MAX_STRING_DIMENSION, quantize.rr_map, quantize.cc_map, quantize.vl_map, quantize.length_map, quantize.img_encoding, quantize.color_weights)
search.iterate()
search.compose_img(preprocess.palette)

plt.imshow(search.string_img)
plt.show()
