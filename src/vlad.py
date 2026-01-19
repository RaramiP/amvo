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


def compute_all_vlad(descriptors_list, vocabulary):
    """
    Calcule les descripteurs VLAD pour toutes les images du jeu de données.

    :param descriptors_list: liste des descripteurs SIFT (un array Nx128 par image)
    :param vocabulary: vocabulaire visuel (Kx128)
    :return: matrice VLAD (n_images x (K*128))
    """
    n_images = len(descriptors_list)
    K, D = vocabulary.shape

    vlad_dim = K * D
    vlad_descriptors = np.zeros((n_images, vlad_dim), dtype=np.float64)

    for i, descriptors in enumerate(descriptors_list):
        if descriptors is None:
            continue  # image sans points d'intérêt

        vlad_descriptors[i] = vlad(descriptors, vocabulary)

    return vlad_descriptors


def reduce_vlad_pca(vlad_vectors, n_components=100):
    """
    Réduction de dimension des VLAD par ACP après centrage.

    :param vlad_vectors: matrice VLAD (n_images x 6400)
    :param n_components: nombre de composantes principales
    :return: VLAD réduits (n_images x n_components)
    """
    # centrage
    mean_vlad = np.mean(vlad_vectors, axis=0)
    vlad_centered = vlad_vectors - mean_vlad

    # ACP
    pca = PCA(n_components=n_components)
    vlad_reduced = pca.fit_transform(vlad_centered)

    return vlad_reduced, pca
