import numpy as np
from sklearn.metrics import pairwise_distances_argmin


def read_stip_file(file):
    keypoints = []
    descriptors = []
    with open(file) as f:
        _ = f.readline()
        for l in f:
            if len(l.strip()) > 0:
                l = l.strip().split(" ")
                keypoints.append([int(k) for k in l[4:9]])
                descriptors.append([float(d) for d in l[9:]])
    if len(descriptors) == 0:
        return None, None
    else:
        return np.array(keypoints, dtype=np.int32), np.array(
            descriptors, dtype=np.float32
        )


def split_descriptors(descriptors, mode="hoghof"):
    if mode == "hog":
        return descriptors[:, :72]
    elif mode == "hof":
        return descriptors[:, 72:]
    elif mode == "hoghof":
        return descriptors
    return None


def compute_bovw(descriptors, vocabulary):
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(len(vocabulary), dtype=np.float32)

    words = pairwise_distances_argmin(descriptors, vocabulary)
    hist, _ = np.histogram(words, bins=len(vocabulary))
    hist = hist.astype(np.float32)

    # normalisation L2
    hist /= np.linalg.norm(hist) + 1e-6
    return hist


def build_bovw_dataset(file_list, vocab, desc_type="hoghof"):
    X = []
    y = []

    for key_file, label in file_list:
        _, descriptors = read_stip_file(key_file)
        descriptors = split_descriptors(descriptors, desc_type)
        bovw = compute_bovw(descriptors, vocab)

        X.append(bovw)
        y.append(label)

    return np.array(X), np.array(y)
