

import os
from glob import glob
import cv2
import matplotlib.pyplot as plt
from aux import *
from read_data import *
from disp import *
from sift_detect import *
import itertools
import csv

cur_kp_1 = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0]])
cur_kp_2 = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0]])

cur_kp_1 = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
cur_kp_2 = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])

cur_kp_1 = np.array([[np.nan, np.nan], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0]])
cur_kp_2 = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0]])

cur_kp_1 = np.array([[np.nan, np.nan], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0]])
cur_kp_2 = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])

cur_kp_1 = np.array([[0, 0] for i in range(8)])
cur_kp_2 = np.array([[0, 0] for i in range(8)])

cur_kp_1 = [[0, 0] for i in range(8)]
cur_kp_2 = [[0, 0] for i in range(8)]

cur_kp_1 = np.array([(-1, 0) for i in range(8)])
cur_kp_2 = np.array([(-100, 0) for i in range(8)])

if cur_kp_1 is None or len(cur_kp_1) < 8 or len(cur_kp_1) != len(cur_kp_2):
    F = np.zeros((3, 3))
else:
    print("Se je zgodilo")
    F, inlier_mask = cv2.findFundamentalMat(cur_kp_1, cur_kp_2, cv2.USAC_MAGSAC, 0.25, 0.99999, 10000)
