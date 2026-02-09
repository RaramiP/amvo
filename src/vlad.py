import numpy as np
import sklearn.neighbors as skln
from sklearn.decomposition import PCA


def vlad(descriptors, vocabulary, use_l2_norm=True, use_sqrt_norm=True):
    """Compute the VLAD descriptors of an image.

    @param sifts SIFT descriptors extracted from an image
    @param vocabulary Visual vocabulary
    @param use_l2_norm True to use global L2 normalization, False otherwise (default: True)
    @param use_sqrt_norm True to use square root normlization, False otherwise (default: True)
    @type sifts Array of shape (N, 128) (N = number of descriptors in the image)
    @type vocabulary Numpy array of shape (K, 128)
    @type use_l2_norm Boolean
    @type use_sqrt_norm Boolean
    @return VLAD vector of the image
    @rtype Numpy array of shape (128*K,)
    """
    vlad = np.zeros(vocabulary.shape, dtype=np.float64)
    quantizer = skln.NearestNeighbors(n_neighbors=1, algorithm="brute").fit(vocabulary)
    ws = quantizer.kneighbors(descriptors, return_distance=False).reshape(-1)

    # compute residuals
    for i in range(len(vlad)):
        if (ws == i).any():
            vlad[i, :] = np.sum(descriptors[ws == i] - vocabulary[i], axis=0)

    # square root normalization
    if use_sqrt_norm:
        vlad[:] = np.sign(vlad) * np.sqrt(np.abs(vlad))

    vlad = vlad.reshape((vlad.shape[0] * vlad.shape[1],))
    if use_l2_norm:
        vlad[:] = vlad / np.maximum(np.linalg.norm(vlad), 1e-12)

    return vlad


def vlad_descriptors(all_descriptors, vocabulary, vlad_func):
    vlad_desc = []
    for desc in all_descriptors:
        if desc is None:
            vlad_vec = np.zeros(vocabulary.shape[0]*vocabulary.shape[1], dtype=np.float32)
        else:
            vlad_vec = vlad_func(desc, vocabulary)
        vlad_desc.append(vlad_vec)
    return np.array(vlad_desc, dtype=np.float32)

def reduce_vlad_dimension(vlad_vectors, n_components):
    pca = PCA(n_components=n_components)
    vlad_centered = vlad_vectors - np.mean(vlad_vectors, axis=0)
    vlad_reduced = pca.fit_transform(vlad_centered)
    return vlad_reduced