

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

def FlattenMatrix(M, num_digits=10):
    return ' '.join([f'{v:.{num_digits}e}' for v in M.flatten()])



file = open('submission.csv', 'w')
#writer = csv.writer(file, quotechar=' ')#quoting=csv.QUOTE_NONE)
#writer.writerow(["sample_id", "fundamental_matrix"])
file.write("sample_id,fundamental_matrix\n")

random.seed(42)
num_features = 2000
src = 'image-matching-challenge-2022'
scenes = find_scenes(src)
sift_detector = cv2.SIFT_create(num_features, contrastThreshold=-10000, edgeThreshold=-10000)

show_images = False
num_show_images = 1
max_pairs_per_scene = 5
max_images_per_scene = 5
verbose = True

test_samples = []
with open(f"{src}/test.csv") as f:
    reader = csv.reader(f, delimiter=",")
    for i, row in enumerate(reader):
        if i==0:
            continue
        test_samples += [row]

#print(test_samples)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

for item in test_samples:

    id1, id2 = item[2], item[3]
    images_dict = {}
    kp_dict = {}
    desc_dict = {}
    scene = item[1]

    #print(f"Processing sample")


    #print('Extracting features...')
    for id in [id1, id2]:
        images_dict[id] = cv2.cvtColor(cv2.imread(f'{src}/test_images/{scene}/{id}.png'), cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(images_dict[id], cv2.COLOR_RGB2GRAY)
        kp, desc = sift_detector.detectAndCompute(gray, None)
        kp_dict[id], desc_dict[id] = kp[:num_features], desc[:num_features]
        print(id)

    #print()


    # Compute matches by brute force.
    cv_matches = bf.match(desc_dict[id1], desc_dict[id2])
    #cv_matches = bf.match(None, None)
    if cv_matches is None:
        cur_kp_1 = None
        cur_kp_2 = None
    else:
        matches = np.array([[m.queryIdx, m.trainIdx] for m in cv_matches])
        cur_kp_1 = ArrayFromCvKps([kp_dict[id1][m[0]] for m in matches])
        cur_kp_2 = ArrayFromCvKps([kp_dict[id2][m[1]] for m in matches])

    print(type(cur_kp_1))
    #print(cur_kp_1)

    if cur_kp_1 is None or len(cur_kp_1) < 8 or len(cur_kp_1) != len(cur_kp_2):
        F = np.zeros((3, 3))
    else:
        # Filter matches with RANSAC.
        # F, inlier_mask = cv2.findFundamentalMat(cur_kp_1, cur_kp_2, cv2.USAC_MAGSAC, 0.25, 0.99999, 10000)
        F, inlier_mask = cv2.findFundamentalMat(cur_kp_1, cur_kp_2, cv2.USAC_MAGSAC, 0.25, 0.99999, 10000)

    if F is None:
        F = np.zeros((3,3))

    tmp = item[0]#f"{s_str};{scene};{id1}-{id2}"
    #tmp2 = np.array2string(np.array(F.flatten()), precision=25, max_line_width=50000, suppress_small=True)[1:-1]
    #writer.writerow([tmp, np.array2string(F.flatten())[1:-1]])
    file.write(f"{tmp},{FlattenMatrix(F)}\n")


file.close()

