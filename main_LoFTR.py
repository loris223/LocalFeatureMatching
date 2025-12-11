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

random.seed(42)

num_features = 2000
src = 'image-matching-challenge-2022/train'
scenes = find_scenes(src)
scaling_dict = read_scaling_dict(src)


show_images = True
num_show_images = 1
max_pairs_per_scene = 20
verbose = True

# We use two different sets of thresholds over rotation and translation. Do not change this -- these are the values used by the scoring back-end.
thresholds_q = np.linspace(1, 10, 10)
thresholds_t = np.geomspace(0.2, 5, 10)

# Save the per-sample errors and the accumulated metric to dictionaries, for later inspection.
errors = {scene: {} for scene in scaling_dict.keys()}
mAA = {scene: {} for scene in scaling_dict.keys()}

matcher = KF.LoFTR(pretrained="outdoor")


for scene in scaling_dict.keys():
    # Load all pairs, find those with a co-visibility over 0.1, and subsample them.
    covisibility_dict = ReadCovisibilityData(f'{src}/{scene}/pair_covisibility.csv')
    pairs = [pair for pair, covis in covisibility_dict.items() if covis >= 0.1]

    print(f'-- Processing scene "{scene}": found {len(pairs)} pairs (will keep {min(len(pairs), max_pairs_per_scene)})',
          flush=True)

    # Subsample the pairs. Note that they are roughly sorted by difficulty (easy ones first), so we shuffle them beforehand: results would be misleading otherwise.
    random.shuffle(pairs)
    pairs = pairs[:max_pairs_per_scene]

    # Extract the images in these pairs (we don't need to load images we will not use).
    ids = []
    for pair in pairs:
        cur_ids = pair.split('-')
        assert cur_ids[0] > cur_ids[1]
        ids += cur_ids
    ids = list(set(ids))

    # Load ground truth data.
    calib_dict = LoadCalibration(f'{src}/{scene}/calibration.csv')

    # Load images and extract SIFT features.
    images_dict = {}
    kp_dict = {}
    desc_dict = {}
    print('Extracting features...')
    for id in tqdm(ids):
        tmp = K.io.load_image(f'{src}/{scene}/images/{id}.jpg', K.io.ImageLoadType.RGB32)[None, ...]
        tmp = K.geometry.resize(tmp, (600, 375), antialias=True)
        tmp.cuda()
        images_dict[id] = tmp
        # images_dict[id] = cv2.cvtColor(cv2.imread(f'{src}/{scene}/images/{id}.jpg'), cv2.COLOR_BGR2RGB)
        # kp_dict[id], desc_dict[id] = ExtractSiftFeatures(images_dict[id], sift_detector, 2000)
    print()
    print(f'Extracted features for {len(kp_dict)} images (avg: {np.mean([len(v) for v in desc_dict.values()])})')

    # Process the pairs.
    max_err_acc_q_new = []
    max_err_acc_t_new = []
    for counter, pair in enumerate(pairs):
        id1, id2 = pair.split('-')

        # Compute matches by brute force.
        #cv_matches = bf.match(desc_dict[id1], desc_dict[id2])
        input_dict = {
            "image0": K.color.rgb_to_grayscale(images_dict[id1]),  # LofTR works on grayscale images only
            "image1": K.color.rgb_to_grayscale(images_dict[id2]),
        }

        with torch.inference_mode():
            correspondences = matcher(input_dict)


        cur_kp_1 = correspondences["keypoints0"].cpu().numpy()
        cur_kp_2 = correspondences["keypoints1"].cpu().numpy()

        #matches = np.array([[m.queryIdx, m.trainIdx] for m in cv_matches])
        #cur_kp_1 = ArrayFromCvKps([kp_dict[id1][m[0]] for m in matches])
        #cur_kp_2 = ArrayFromCvKps([kp_dict[id2][m[1]] for m in matches])

        # Filter matches with RANSAC.
        F, inlier_mask = cv2.findFundamentalMat(cur_kp_1, cur_kp_2, cv2.USAC_MAGSAC, 0.25, 0.99999, 10000)
        inlier_mask = inlier_mask.astype(bool).flatten()

        #matches_after_ransac = np.array([match for match, is_inlier in zip(matches, inlier_mask) if is_inlier])
        ##inlier_kp_1 = ArrayFromCvKps([kp_dict[id1][m[0]] for m in matches_after_ransac])
        #inlier_kp_2 = ArrayFromCvKps([kp_dict[id2][m[1]] for m in matches_after_ransac])

        inlier_kp_1 = [kp for kp, is_inlier in zip(cur_kp_1, inlier_mask) if is_inlier]
        inlier_kp_2 = [kp for kp, is_inlier in zip(cur_kp_2, inlier_mask) if is_inlier]

        # Compute the essential matrix.
        E, R, T = ComputeEssentialMatrix(F, calib_dict[id1].K, calib_dict[id2].K, inlier_kp_1, inlier_kp_2)
        q = QuaternionFromMatrix(R)
        T = T.flatten()

        # Get the relative rotation and translation between these two cameras, given their R and T in the global reference frame.
        R1_gt, T1_gt = calib_dict[id1].R, calib_dict[id1].T.reshape((3, 1))
        R2_gt, T2_gt = calib_dict[id2].R, calib_dict[id2].T.reshape((3, 1))
        dR_gt = np.dot(R2_gt, R1_gt.T)
        dT_gt = (T2_gt - np.dot(dR_gt, T1_gt)).flatten()
        q_gt = QuaternionFromMatrix(dR_gt)
        q_gt = q_gt / (np.linalg.norm(q_gt) + eps)

        # Compute the error for this example.
        err_q, err_t = ComputeErrorForOneExample(q_gt, dT_gt, q, T, scaling_dict[scene])
        errors[scene][pair] = [err_q, err_t]

        tmp_N = len(cur_kp_1)
        tmp_subsample = random.sample(range(tmp_N), int(tmp_N / 5))

        mkpts0 = cur_kp_1
        mkpts1 = cur_kp_2
        tmp_arr = [i for i, x in enumerate(inlier_mask) if x]
        tmp_sample = random.sample(tmp_arr, 10)
        inliers = np.zeros(inlier_mask.shape, dtype=np.bool_)
        inliers[tmp_sample] = True
        #inlier_mask = [p for p in inlier_mask]

        # Plot the resulting matches and the pose error.
        if verbose or (show_images and counter < num_show_images):
            print(f'{pair}, err_q={(err_q):.02f} (deg), err_t={(err_t):.02f} (m)', flush=True)
        if show_images and counter < num_show_images or err_t > 0:
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
                K.tensor_to_image(images_dict[id1]),
                K.tensor_to_image(images_dict[id2]),
                inliers,
                draw_dict={"inlier_color": (0.2, 1, 0.2), "tentative_color": None, "feature_color": (0.2, 0.5, 1),
                           "vertical": False},
            )
            plt.show()
            print()

    # Histogram the errors over this scene.
    mAA[scene] = ComputeMaa([v[0] for v in errors[scene].values()], [v[1] for v in errors[scene].values()],
                            thresholds_q, thresholds_t)
    print()
    print(f'Mean average Accuracy on "{scene}": {mAA[scene][0]:.05f}')
    print()
    break

print()
print('------- SUMMARY -------')
print()
for scene in scaling_dict.keys():
    print(f'-- Mean average Accuracy on "{scene}": {mAA[scene][0]:.05f}')
print()
print(f'Mean average Accuracy on dataset: {np.mean([mAA[scene][0] for scene in mAA]):.05f}')