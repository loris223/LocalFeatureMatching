import matplotlib.pyplot as plt
import cv2

# Functions to plot



def disp_multiple_pics(images_dict):
    num_rows = 6
    num_cols = 4
    f, axes = plt.subplots(num_rows, num_cols, figsize=(20, 20), constrained_layout=True)
    for i, key in enumerate(images_dict):
        if i >= num_rows * num_cols:
            break
        cur_ax = axes[i % num_rows, i // num_rows]
        cur_ax.imshow(images_dict[key])
        cur_ax.set_title(key)
        cur_ax.axis('off')
    plt.show()


def disp_covisibility_sample(covisibility_dict, images_dict):
    # Let's look at easy pairs first, and difficult pairs later.
    easy_subset = [k for k, v in covisibility_dict.items() if v >= 0.7]
    difficult_subset = [k for k, v in covisibility_dict.items() if v >= 0.1 and v < 0.2]

    for i, subset in enumerate([easy_subset, difficult_subset]):
        print(f'Pairs from an {"easy" if i == 0 else "difficult"} subset')

        for pair in subset[:4]:
            # A pair string is simply two concatenated image IDs, separated with a hyphen.
            image_id_1, image_id_2 = pair.split('-')

            f, axes = plt.subplots(1, 2, figsize=(15, 10), constrained_layout=True)
            axes[0].imshow(images_dict[image_id_1])
            axes[0].set_title(image_id_1)
            axes[1].imshow(images_dict[image_id_2])
            axes[1].set_title(image_id_2)
            for ax in axes:
                ax.axis('off')
            plt.show()

        print()
        print()

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    plt.title('Covisibility histogram')
    plt.hist(list(covisibility_dict.values()), bins=10, range=[0, 1])
    plt.show()

def disp_keypoints(image, keypoints):
    # Each local feature contains a keypoint (xy, possibly scale, possibly orientation) and a description vector (128-dimensional for SIFT).
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, outImage=None,
                                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    fig = plt.figure(figsize=(15, 15))
    plt.imshow(image_with_keypoints)
    plt.axis('off')
    plt.show()