import cv2
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import torch
from kornia_moons.viz import draw_LAF_matches


import os
from glob import glob
import cv2
import matplotlib.pyplot as plt
from aux import *
from read_data import *
from disp import *
from sift_detect import *


num_features = 100
src = 'image-matching-challenge-2022/train'
scenes = find_scenes(src)
scene = scenes[8]
scaling_dict = read_scaling_dict(src)

covisibility_dict = ReadCovisibilityData(f'{src}/{scene}/pair_covisibility.csv')
pairs = [pair for pair, covis in covisibility_dict.items() if covis >= 0.1]

print(pairs[0])
pair = pairs[0]
cur_ids = pair.split('-')
id_1, id_2 = cur_ids[0], cur_ids[1]
image_1 = cv2.cvtColor(cv2.imread(f'{src}/{scene}/images/{id_1}.jpg'), cv2.COLOR_BGR2RGB)
image_2 = cv2.cvtColor(cv2.imread(f'{src}/{scene}/images/{id_2}.jpg'), cv2.COLOR_BGR2RGB)

img1 = K.io.load_image(f'{src}/{scene}/images/{id_1}.jpg', K.io.ImageLoadType.RGB32)[None, ...]
img2 = K.io.load_image(f'{src}/{scene}/images/{id_2}.jpg', K.io.ImageLoadType.RGB32)[None, ...]

img1 = K.geometry.resize(img1, (600, 375), antialias=True)
img2 = K.geometry.resize(img2, (600, 375), antialias=True)


matcher = KF.LoFTR(pretrained="outdoor")

input_dict = {
    "image0": K.color.rgb_to_grayscale(img1),  # LofTR works on grayscale images only
    "image1": K.color.rgb_to_grayscale(img2),
}

with torch.inference_mode():
    correspondences = matcher(input_dict)

mkpts0 = correspondences["keypoints0"].cpu().numpy()
mkpts1 = correspondences["keypoints1"].cpu().numpy()
print(mkpts0)
print(mkpts1)
Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
inliers = inliers > 0

print(inliers)
selector_vec = np.random.choice(len(mkpts0), size=200, replace=False)
#mkpts0 = mkpts0[:100]
#mkpts1 = mkpts1[:100]
#inliers = inliers[:100]
mkpts0 = mkpts0[selector_vec]
mkpts1 = mkpts1[selector_vec]
inliers = inliers[selector_vec]
draw_LAF_matches(
    KF.laf_from_center_scale_ori(
        torch.from_numpy(mkpts0).view(1, -1, 2),
        torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
        torch.ones(mkpts0.shape[0]).view(1, -1, 1),
    ),
    KF.laf_from_center_scale_ori(
        torch.from_numpy(mkpts1).view(1, -1, 2),
        torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
        torch.ones(mkpts1.shape[0]).view(1, -1, 1),
    ),
    torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
    K.tensor_to_image(img1),
    K.tensor_to_image(img2),
    inliers,
    draw_dict={"inlier_color": (0.2, 1, 0.2), "tentative_color": None, "feature_color": (0.2, 0.5, 1), "vertical": False},
)
plt.show()