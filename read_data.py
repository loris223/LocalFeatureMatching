import os
import cv2
from glob import glob
from aux import ReadCovisibilityData
import csv
import numpy as np
from collections import namedtuple


Gt = namedtuple('Gt', ['K', 'R', 'T'])

def LoadCalibration(filename):
    '''Load calibration data (ground truth) from the csv file.'''

    calib_dict = {}
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            # Skip header.
            if i == 0:
                continue

            camera_id = row[0]
            K = np.array([float(v) for v in row[1].split(' ')]).reshape([3, 3])
            R = np.array([float(v) for v in row[2].split(' ')]).reshape([3, 3])
            T = np.array([float(v) for v in row[3].split(' ')])
            calib_dict[camera_id] = Gt(K=K, R=R, T=T)

    return calib_dict


def find_scenes(src):
    val_scenes = []
    for f in os.scandir(src):
        if f.is_dir():
            cur_scene = os.path.split(f)[-1]
            # print(f'Found scene "{cur_scene}"" at {f.path}')
            val_scenes += [cur_scene]
    return val_scenes


def read_images_names(src, scene):
    images_names = []
    for filename in glob(f'{src}/{scene}/*.png'):
        images_names.append(os.path.basename(os.path.splitext(filename)[0]))
    return images_names

def read_images(src, scene):
    images_dict = {}
    for filename in glob(f'{src}/{scene}/images/*.jpg'):
        cur_id = os.path.basename(os.path.splitext(filename)[0])

        # OpenCV expects BGR, but the images are encoded in standard RGB, so you need to do color conversion if you use OpenCV for I/O.
        images_dict[cur_id] = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    return images_dict

def read_covisibility_data(src, scene):
    covisibility_dict = ReadCovisibilityData(f'{src}/{scene}/pair_covisibility.csv')
    return covisibility_dict


def read_calibration_dict(src, scene):
    calib_dict = LoadCalibration(f'{src}/{scene}/calibration.csv')
    #print(f'Loded ground truth data for {len(calib_dict)} images')
    #print()
    return calib_dict


def read_scaling_dict(src):
    scaling_dict = {}
    with open(f'{src}/scaling_factors.csv') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            # Skip header.
            if i == 0:
                continue
            scaling_dict[row[0]] = float(row[1])

    #print(f'Scaling factors: {scaling_dict}')
    #print()
    return scaling_dict