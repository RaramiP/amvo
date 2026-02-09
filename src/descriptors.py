import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

def color_histogram(im, bins_per_channel=8):
    im = im.copy()
    bin_width = 256.0 / bins_per_channel
    im = (im / bin_width).astype(np.uint32)
    im = im[..., 0] * bins_per_channel**2 + im[..., 1] * bins_per_channel + im[..., 2]
    hist = np.zeros((bins_per_channel**3,), dtype=np.float32)
    colors, counts = np.unique(im, return_counts=True)
    hist[colors] = counts
    hist = hist / np.linalg.norm(hist, ord=1)
    return hist

def compute_color_descriptors(image_files, bins_per_channel=8):
    desc = []
    for file in image_files:
        img = cv2.imread(file)
        desc.append(color_histogram(img, bins_per_channel=bins_per_channel))
    return np.array(desc, dtype=np.float32)


def lbp_histogram(im, P=8, R=1, nbins=256):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P, R, method="default")
    hist, _ = np.histogram(lbp.ravel(), bins=nbins, range=(0, nbins))
    hist = hist.astype(np.float32)
    hist = hist / np.linalg.norm(hist, ord=1)
    return hist

def lbp_descriptors(image_files, P=8, R=1, nbins=256):
    desc = []
    for file in image_files:
        img = cv2.imread(file)
        desc.append(lbp_histogram(img, P=P, R=R, nbins=nbins))
    return np.array(desc, dtype=np.float32)

def fuse_descriptors(*desc_list):
    return np.hstack(desc_list)

def grid_descriptor(im, descriptor="color", grid_size=(5, 5), bins_per_channel=8, P=8, R=1, nbins=256):
    h, w = im.shape[:2]
    dh, dw = h // grid_size[0], w // grid_size[1]
    region_desc = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            region = im[i*dh:(i+1)*dh, j*dw:(j+1)*dw]
            if descriptor == "color":
                hist = color_histogram(region, bins_per_channel=bins_per_channel)
            elif descriptor == "lbp":
                hist = lbp_histogram(region, P=P, R=R, nbins=nbins)
            region_desc.append(hist)
    return np.hstack(region_desc)

def grid_descriptors(image_files, descriptor="color", grid_size=(5,5), bins_per_channel=8, P=8, R=1, nbins=256):
    desc = []
    for file in image_files:
        img = cv2.imread(file)
        desc.append(grid_descriptor(img, descriptor=descriptor, grid_size=grid_size,
                                    bins_per_channel=bins_per_channel, P=P, R=R, nbins=nbins))
    return np.array(desc, dtype=np.float32)

def train_test(descriptors, labels, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(descriptors, labels, test_size=test_size, random_state=random_state, stratify=labels)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=sorted(set(labels)))
    return acc, cm, y_test, y_pred

def sift_descriptors(image_files):
    sift = cv2.SIFT_create()
    all_keypoints = []
    all_descriptors = []
    for file in image_files:
        img = cv2.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        all_keypoints.append(keypoints)
        all_descriptors.append(descriptors)
    return all_keypoints, all_descriptors

def draw_sift_keypoints(img, keypoints):
    img_kp = img.copy()
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(img_kp, (x, y), int(round(kp.size/2)), (0, 255, 0), 1)
    return img_kp

def bow_histogram(descriptors, vocabulary):
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(vocabulary.shape[0], dtype=np.float32)
    nn = NearestNeighbors(n_neighbors=1, algorithm='brute').fit(vocabulary)
    distances, indices = nn.kneighbors(descriptors)
    hist = np.zeros(vocabulary.shape[0], dtype=np.float32)
    for idx in indices.flatten():
        hist[idx] += 1
    hist = hist / np.linalg.norm(hist, ord=1)
    return hist

def bow_descriptors(all_descriptors, vocabulary):
    bow_desc = []
    for desc in all_descriptors:
        bow_desc.append(bow_histogram(desc, vocabulary))
    return np.array(bow_desc, dtype=np.float32)
