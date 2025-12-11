import cv2
import matplotlib.pyplot as plt

def ExtractSiftFeatures(image, detector, num_features):
    '''Compute SIFT features for a given image.'''

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kp, desc = detector.detectAndCompute(gray, None)
    return kp[:num_features], desc[:num_features]

def sift_get_local_features(image, num_features=5000):

    # You may want to lower the detection threshold, as small images may not be able to reach the budget otherwise.
    # Note that you may actually get more than num_features features, as a feature for one point can have multiple orientations (this is rare).
    sift_detector = cv2.SIFT_create(num_features, contrastThreshold=-10000, edgeThreshold=-10000)


    keypoints, descriptors = ExtractSiftFeatures(image, sift_detector, num_features)
    #print(f'Computed {len(keypoints)} features.')

    return keypoints, descriptors

