import cv2
import numpy as np
import operator
from keras.models import load_model, model_from_json
import sudoku-solver as sol

classifier = load_model("./digit_model.h5")

margin = 4
box = 28 + 2 *margin
grid_size = 9 * box

vid = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
flag = 0
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (1080, 620))
