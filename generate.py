import os
from matplotlib import pyplot as plt

from src.preprocess import PreProcessManager
from src.calculate import CalculateManager
from src.search import SearchManager
os.system('cls||clear')

NUM_COLORS = 6
NUM_POINTS = 256
NUM_EDGE_LINES = 750
NUM_COLOR_LINES = 2000


MAX_LOAD_DIMENSION = 1024
MAX_STRING_DIMENSION = 4096
PATH = 'images/girl_with_pearl_earring.jpg'
OUTPUT_FOLDER = 'results/'

BG_FLAG = 0
BG_THRESHOLD = 0
IMG_OFFSET_X = 25
IMG_OFFSET_Y = -50
IMG_COVERAGE = 0.9
GAUSSIAN_SIGMA = 2
MONTE_CARLO_CHOICES = 5000


preprocess = PreProcessManager()
preprocess.load_image(PATH, MAX_LOAD_DIMENSION)
preprocess.gaussian_edge(GAUSSIAN_SIGMA)
preprocess.circle_crop(IMG_COVERAGE, IMG_OFFSET_X, IMG_OFFSET_Y)
preprocess.reduce_colors(NUM_COLORS, OUTPUT_FOLDER)
preprocess.dither(NUM_COLORS)
preprocess.resize_canvas(MONTE_CARLO_CHOICES, MAX_STRING_DIMENSION)

calculate = CalculateManager()
calculate.compute_pin_mapping(preprocess.radius, MAX_STRING_DIMENSION, NUM_POINTS)
calculate.analyze_img(preprocess.img, preprocess.palette)

search = SearchManager()
search.init_globals(NUM_COLORS, NUM_POINTS, NUM_EDGE_LINES, NUM_COLOR_LINES, MAX_STRING_DIMENSION)
search.create_GPU_objects(calculate.rr_map, calculate.cc_map, calculate.vl_map, calculate.length_map, calculate.normalizer_map, preprocess.edge, calculate.img_encoding, calculate.color_weights)
search.draw_color()
search.draw_edge()
search.compose_img(preprocess.palette, OUTPUT_FOLDER)
# search.generate_chart(preprocess.palette, OUTPUT_FOLDER)

plt.imshow(search.string_img)
plt.show()