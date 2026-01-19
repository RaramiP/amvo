import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from skimage.feature import local_binary_pattern
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression

from typing import Tuple

DATA_PATH = "../data/caltech101_subset/"
SUBSET_PATH = DATA_PATH + "caltech101_subset.files"
SEED = 42
np.random.seed(SEED)


def color_histogram(im, bins_per_channel=8):
    """Computes a joint color histogram.
    :param im Color image as a Numpy array of shape (height, width, 3)
    @param bins_per_channel Number of bins per channel after quantization
    @type im Numpy array of type uint8 and shape (height, width, 3)
    @type bins_per_channel Integer
    @return Normalized color histogram
    @rtype Numpy array of type float32 and shape (bins_per_channel**3,)
    """
    im = im.copy()

    # quantize image
    bin_width = 256.0 / bins_per_channel
    im = (im / bin_width).astype(np.uint32)

    # flatten color space
    im = im[..., 0] * bins_per_channel**2 + im[..., 1] * bins_per_channel + im[..., 2]

    # compute and normalize histogram
    histogram = np.zeros((bins_per_channel**3,), dtype=np.float32)
    colors, counts = np.unique(im, return_counts=True)
    histogram[colors] = counts
    histogram = histogram / np.linalg.norm(histogram, ord=1)
    return histogram


def load_image(grey=False):
    """Loads images and their labels from a subset file.
    :return List of loaded images and corresponding labels
    @rtype (list of Numpy arrays, Numpy array)
    """
    if not os.path.exists(SUBSET_PATH) and not os.path.isfile(SUBSET_PATH):
        raise FileNotFoundError(f"{SUBSET_PATH} not found")

    with open(SUBSET_PATH, "r") as f:
        lines = f.readlines()

    imgs, labels = [], []
    for line in lines:
        split = line.split()
        if len(split) != 2:
            continue

        img_path, label = split
        if not os.path.exists(os.path.join(DATA_PATH, img_path)):
            continue

        img = cv2.imread(os.path.join(DATA_PATH, img_path))
        if grey:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgs.append(img)
        labels.append(label)
    return imgs, np.array(labels)


def histogram(imgs):
    """Computes color histograms for a set of images.
    :param imgs List of color images
    @type imgs List of Numpy arrays of shape (height, width, 3)
    :return List of normalized color histograms
    @rtype List of Numpy arrays
    """
    hists = []
    for img in imgs:
        hist = color_histogram(img)
        hists.append(hist)
    return hists


def LBP(imgs):
    """Computes Local Binary Pattern (LBP) histograms for images.
    :param imgs List of color images
    @type imgs List of Numpy arrays of shape (height, width, 3)
    :return List of normalized LBP histograms
    @rtype List of Numpy arrays
    """
    res = []
    for img in imgs:
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        local = local_binary_pattern(grey, 8, 1)
        n_bin = int(local.max() + 1)
        hist = np.histogram(local, bins=np.arange(n_bin + 1), density=True)[0]
        res.append(hist)
    return res


def geometric_split(imgs, function, nb_split=5):
    """Applies a geometric split and feature extraction on images.
    :param imgs List of images
    :param function Feature extraction function applied to each region
    :param nb_split Number of geometric splits
    @type imgs List of Numpy arrays
    @type function Callable
    @type nb_split Integer
    :return List of concatenated feature vectors
    @rtype List of Numpy arrays
    """
    res = []
    for img in imgs:
        hist = []
        for i in range(nb_split):
            shape = (img.shape[0] // nb_split, img.shape[1] // nb_split)
            hist.append(
                function(
                    [
                        img[
                            i * shape[0] : (i + 1) * shape[0],
                            i * shape[1] : (i + 1) * shape[1],
                        ]
                    ]
                )[0]
            )
        res.append(np.hstack(hist))
    return res


def process_train_test(data, labels):
    """Trains a model and evaluates it on a test set.
    :param model Machine learning model
    :param data Feature vectors
    :param labels Ground truth labels
    @type model Object with fit and predict methods
    @type data Numpy array
    @type labels Numpy array
    :return Accuracy score, confusion matrix, and labels
    @rtype (float, Numpy array, Numpy array)
    """
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
    model = LogisticRegression(max_iter=1000, multi_class="auto", solver="lbfgs")
    model.fit(X_train, y_train)
    predic = model.predict(X_test)
    return accuracy_score(y_test, predic), confusion_matrix(predic, y_test)


def SIFT_create(imgs):
    """Computes SIFT keypoints and descriptors for a set of images.
    :param imgs List of images
    @type imgs List of Numpy arrays of shape (height, width, 3)
    :return List of SIFT keypoints and descriptors
    @rtype (list of list of cv2.KeyPoint, list of Numpy arrays)
    """
    keypoints = []
    descriptors = []
    sift = cv2.SIFT_create()
    for img in imgs:
        keypoint, desc = sift.detectAndCompute(img, None)
        keypoints.append(keypoint)
        descriptors.append(desc)
    return keypoints, descriptors


def draw_keypoint(img: np.ndarray, keypoints: cv2.KeyPoint):
    """Draws keypoints on an image.
    :param img Image to draw keypoints on
    :param keypoints List of keypoints
    @type img Numpy array of type uint8 and shape (height, width, 3)
    @type keypoints List of cv2.KeyPoint
    """
    for kp in keypoints:
        x, y = np.round(kp.pt).astype(int)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
    plt.imshow(img)
    plt.show()


def draw_keypoint_with_cv(img: np.ndarray, keypoints: Tuple[cv2.KeyPoint]):
    """Draws keypoints on an image.
    :param img Image to draw keypoints on
    :param keypoints List of keypoints
    @type img Numpy array of type uint8 and shape (height, width, 3)
    @type keypoints List of cv2.KeyPoint
    """
    out_img = cv2.drawKeypoints(
        img,
        keypoints,
        None,
        color=(0, 0, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    cv2.imshow("Keypoints", out_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visual_word(descriptors, vocabulary):
    if descriptors is None:
        return np.zeros(vocabulary.shape[0])

    nn = NearestNeighbors(n_neighbors=1, algorithm="brute")
    nn.fit(vocabulary)
    indices = nn.kneighbors(descriptors, return_distance=False)

    hist = np.zeros(vocabulary.shape[0])
    for idx in indices:
        hist[idx[0]] += 1
    return hist


def histograms_from_descriptors(descriptors_list, vocabulary):
    n_images = len(descriptors_list)
    n_words = vocabulary.shape[0]
    hists = np.zeros((n_images, n_words))

    for i, desc in enumerate(descriptors_list):
        hists[i] = visual_word(desc, vocabulary)

    return hists
