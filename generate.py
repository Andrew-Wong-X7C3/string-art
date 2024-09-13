import os
import numpy as np
from matplotlib import pyplot as plt

from src.preprocess import PreProcessManager
from src.calculate import CalculateManager
from src.search import SearchManager

os.system('cls||clear')

NUM_COLORS = 8
NUM_POINTS = 256
NUM_LINES = 8000

MAX_LOAD_DIMENSION = 1024
MAX_STRING_DIMENSION = 4096
PATH = 'images/girl_with_pearl_earring.jpg'
OUTPUT_FOLDER = 'results/'

BG_FLAG = 0
BG_THRESHOLD = 0
IMG_OFFSET_X = 25
IMG_OFFSET_Y = -50
IMG_COVERAGE = 0.9
MONTE_CARLO_CHOICES = 1000


preprocess = PreProcessManager()
preprocess.load_image(PATH, MAX_LOAD_DIMENSION)
preprocess.circle_crop(IMG_COVERAGE, IMG_OFFSET_X, IMG_OFFSET_Y)
preprocess.reduce_colors(NUM_COLORS, OUTPUT_FOLDER)
preprocess.dither(NUM_COLORS)
preprocess.resize_canvas(MONTE_CARLO_CHOICES, MAX_STRING_DIMENSION)

calculate = CalculateManager()
calculate.compute_pin_mapping(preprocess.radius, MAX_STRING_DIMENSION, NUM_POINTS)
# import pickle
# with open(r'C:\Users\Andrew\Documents\GitHub\string-art-old\maps_2046.pkl', 'rb') as f:
#     calculate.rr_map, calculate.cc_map, calculate.vl_map, calculate.length_map =  pickle.load(f)
calculate.analyze_img(preprocess.img, preprocess.palette)

search = SearchManager()
search.init_globals(NUM_COLORS, NUM_POINTS, NUM_LINES, MAX_STRING_DIMENSION)
search.create_GPU_objects(MAX_STRING_DIMENSION, calculate.rr_map, calculate.cc_map, calculate.vl_map, calculate.length_map, calculate.img_encoding, calculate.color_weights)
search.iterate()
search.compose_img(preprocess.palette, OUTPUT_FOLDER)
search.generate_chart(preprocess.palette, OUTPUT_FOLDER)

plt.imshow(search.string_img)
plt.show()
