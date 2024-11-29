import jax
import jax.numpy as jnp
import numpy as np
import jraph
import optax
import pickle

import matplotlib.pyplot as plt
from matplotlib import colormaps
from mpl_toolkits.axes_grid1 import make_axes_locatable

from toy_examples.example_HGs import tetrahedron, OH_rectangle
from morphomatics.graph.operators import max_pooling, mean_pooling
from morphomatics.nn import MfdInvariant
import random
from OrientedHypergraphs.utils import pickle_load
from OrientedHypergraphs.objects import OrientedHypergraph, OHtuple_to_jraph_edgeless, jraph_to_OH, OHGraphTuple, \
    OrientedHypergraphTuple_e2v_struct


def artificialIn_SAR_image(pts, sigma=0.3, save=False):
    """
    Generate an artificial SAR image is in ManoptExamples.jl
    :param pts: Dimension of the SAR image as pts x pts
    :param sigma: Variance parameter of the Gaussian noice added to the image
    :param save: Whether to save the image with pickle.
    :return:Image as pts x pts array of points on S1
    """
    # variables
    # rotation of ellipse
    aEll = np.radians(35.0)
    cosE = np.cos(aEll)
    sinE = np.sin(aEll)
    aStep = np.radians(45.0)
    cosA = np.cos(aStep)
    sinA = np.sin(aStep)
    # main and minor axis of the ellipse
    axes_inv = [6, 25]
    # values for the hyperboloid
    mid_point = [0, 0.375]
    # mid_point = [0, 0.275]#[0.275, 0.275]
    radius = 0.15
    values = np.linspace(-0.5, 0.5, pts)
    # values = [range(-0.5, 0.5; length=pts)...]
    # Steps
    aSteps = np.radians(60.0)
    cosS = np.cos(aSteps)
    sinS = np.sin(aSteps)
    l = 0.075
    midP = [-0.475, -0.0625]  # .125, .55]
    # midP = [.125, .055]
    img = np.zeros((pts, pts))
    img_noisy = np.zeros((pts, pts))
    # for j in eachindex(values), i in eachindex(values)
    for j in range(len(values)):
        for i in range(len(values)):
            # ellipse
            Xr = cosE * values[i] - sinE * values[j]
            Yr = cosE * values[j] + sinE * values[i]
            v = axes_inv[0] * Xr ** 2 + axes_inv[1] * Yr ** 2
            if v <= 1.0:
                k1 = 10.0 * np.pi * Yr
            else:
                k1 = 0.0
            # k1 = v <= 1.0 ? 10.0 * np.pi * Yr : 0.0
            # circle
            Xr = cosA * values[i] - sinA * values[j]
            Yr = cosA * values[j] + sinA * values[i]
            v = ((Xr - mid_point[0]) ** 2 + (Yr - mid_point[1]) ** 2) / radius ** 2
            if v <= 1.0:
                k2 = -4.0 * np.pi * (1.0 - v)
            else:
                k2 = 0.0
            # k2 = v <= 1.0 ? 4.0 * np.pi * (1.0 - v) : 0.0
            #
            Xr = cosS * values[i] - sinS * values[j]
            Yr = cosS * values[j] + sinS * values[i]
            k3 = 0.0
            for m in range(1, 9):
                # in_range = (abs(Xr + midP[1] + m * l) + abs(Yr + midP[2] + m * l)) ≤ l
                # k3 += in_range ? 2 * pi * (m / 8) : 0.0
                in_range = abs(Xr + midP[0] + m * l) + abs(Yr + midP[1] + m * l)
                if in_range <= l:
                    k3 -= 2 * np.pi * (m / 8)
                else:
                    k3 += 0.0
            img[j, i] = np.mod(k1 + k2 + k3 - np.pi, 2 * np.pi) + np.pi
            # noice = np.random.normal(0, sigma)
            # img_noisy[j, i] = (k1 + k2 + k3 - np.pi + noice) % (2 * np.pi) + np.pi
            if k1 + k2 + k3 - np.pi < -np.pi:
                pass
            img_noisy[j, i] = np.mod(k1 + k2 + k3 - np.pi + np.random.normal(0, sigma), 2 * np.pi) + np.pi
            # img[j, i] = k1 + k2 + k3 + np.pi
            # img_noisy[j, i] = k1 + k2 + k3 + np.random.normal(0, sigma) + np.pi

    if save:
        jnp.save(f"phase_img_{pts}", img, allow_pickle=True, fix_imports=True)
        jnp.save(f"phase_img_{pts}_noisy", img, allow_pickle=True, fix_imports=True)
    return img, img_noisy


def graphify_image(img, hyperedges=False, ForwardOH=False, BackwardOH=False):
    """

    :param img: Image to make oriented local hypergraphs out of
    :param hyperedges: if True then the F - and B-hyperedges are returned instead of graphedges. NB: by default empty
    :param ForwardOH: Add F-hyperedges to edges, requires hyperedges=True
    :param BackwardOH: Add B-hyperedges to edges, requires hyperedges=True
    :return: e2v_in and e2v_out sets defining an oriented hyperedge set
    """
    e2v_in, e2v_out = [], []
    e2v_in_hyp, e2v_out_hyp = [], []
    l_x, l_y = len(img), len(img[0])
    for i in range(l_x):
        for j in range(l_y):
            for s in (-1, 1):
                e2v_in.extend((i + j * l_y, i + j * l_y))
                e2v_out.append((i + s) % l_x + j * l_y)
                e2v_out.append(i + (j + s) % l_y * l_y)
            if hyperedges:
                if ForwardOH:
                    e2v_in_hyp.append([e2v_in[-1]])
                    e2v_out_hyp.append(e2v_out[-4:])
                if BackwardOH:
                    e2v_in_hyp.append(e2v_out[-4:])
                    e2v_out_hyp.append([e2v_in[-1]])

    if hyperedges:
        return e2v_in_hyp, e2v_out_hyp
    return np.array(e2v_in)[..., None], np.array(e2v_out)[..., None]


def graphify_image_non_local(img, k_nearest=5, patch_size=1, connect_middle=False, no_overlap=True):
    """
    Makes hypergraphs out of an image based on k-nearest neighbors of patches in featurespace
    :param img: Image to make oriented local hypergraphs out of
    :param k_nearest:
    :param patch_size:
    :param connect_middle:
    :param no_overlap:
    :return:
    """
    e2v_in, e2v_out, W = [], [], []
    senders, receivers = [], []
    l_x, l_y = len(img), len(img[0])
    D_ind = np.zeros((l_x, l_y, (1 + 2 * patch_size) ** 2, 2), dtype=int)
    for i in range(l_x):
        for j in range(l_y):
            Dij = []
            for s_x in range(-patch_size, patch_size + 1):
                for s_y in range(-patch_size, patch_size + 1):
                    Dij.append(((i + s_x) % (l_x), (j + s_y) % (l_y)))
            D_ind[i, j] = np.asarray(Dij, dtype=int)
    D = img[(D_ind[:, :, ..., 0], D_ind[:, :, ..., 1])]

    def S1_distance(s_1, s_2):
        return np.min([abs(s_1 - s_2), 2 * np.pi - abs(s_1 - s_2)], axis=0)

    def matrix_to_stack_index(indexes, l_x=l_x, l_y=l_y):
        assert indexes.shape[-1] == 1 or indexes.shape[-1] == 2, "indexes should have shape where last axis is 1 or 2"
        if indexes.shape[-1] == 1:
            indexes = jnp.asarray([indexes[..., 0] % l_x, indexes[..., 0] // l_y])
        else:
            indexes = jnp.asarray([indexes[..., 0] + l_y * indexes[..., 1]])
        indexes = jnp.rollaxis(indexes, 0, len(indexes.shape))
        return indexes

    D_ind_flat = matrix_to_stack_index(D_ind)
    print("Beginning finding k nearest feature patches")
    for i in range(l_x):
        for j in range(l_y):
            D_distances = np.zeros((l_x, l_y))
            #   TODO: This for-loop comparing of distances is slow
            for i_comp in range(l_x):
                for j_comp in range(l_y):
                    if no_overlap:
                        if abs(i_comp - i) <= 2 * patch_size and abs(j_comp - j) <= 2 * patch_size:
                            D_distances[i_comp, j_comp] = np.infty
                        else:
                            D_distances[i_comp, j_comp] = np.linalg.norm(S1_distance(D[i, j], D[i_comp, j_comp]))
                    if i_comp == i and j_comp == j:
                        D_distances[i_comp, j_comp] = np.infty
                    else:
                        D_distances[i_comp, j_comp] = np.linalg.norm(S1_distance(D[i, j], D[i_comp, j_comp]))
            k_nearest_i = jnp.argpartition(D_distances.flatten(), kth=k_nearest)[0:k_nearest]
            k_nearest_ij = matrix_to_stack_index(k_nearest_i[..., None])
            # k_nearest_patches = D_ind_flat[i, j].reshape((1+2*patch_size)**2)[k_nearest_ij]
            k_nearest_patches = D_ind_flat[k_nearest_ij[:, 0], k_nearest_ij[:, 1]][:, :, 0]
            for k, D_k in enumerate(k_nearest_patches):
                edge_in = D_ind_flat[i, j, :, 0]
                edge_out = D_k
                e2v_in.append(edge_in)
                e2v_in.append(edge_out)
                e2v_out.append(edge_out)
                e2v_out.append(edge_in)
                senders.append(matrix_to_stack_index(jnp.array([i, j])))
                senders.append(edge_out[len(edge_in) // 2][None])
                receivers.append(edge_out[len(edge_in) // 2][None])
                receivers.append(matrix_to_stack_index(jnp.array([i, j])))
                w_scaling = jnp.sqrt((k_nearest - k)/k_nearest)*D_distances.flatten()[k_nearest_i[0]]
                if w_scaling == 0:
                    w_scaling = 1
                    print("w_scaling off")
                W.append([w_scaling/D_distances.flatten()[k_nearest_i[k]]])
                W.append([w_scaling/D_distances.flatten()[k_nearest_i[k]]])  # Attempt also linear interpolation
            if (i * l_x + j) % (l_x*l_y // 10) == 0:
                print(f"{i * l_x + j} of {l_x * l_y} pixels found k={k_nearest} nearest of")

    if connect_middle:
        return senders, receivers, W
    return e2v_in, e2v_out, W


def phase_image_OH(pts, return_image=False, sigma=0.2, non_local=True, k_nearest=5, patch_size=1, connect_middle=False,
                   hyper_local=False, no_overlap=True, ForwardOH=False, BackwardOH=False):
    img, img_noisy = artificialIn_SAR_image(pts, sigma)
    print("S1 image generated")
    # plt.imshow(img, cmap="hsv")
    # plt.show()
    # plt.imshow(img_noisy, cmap="hsv")
    # plt.show()
    N_pixels = len(img.flatten())
    if non_local:
        e2v_in, e2v_out, W = graphify_image_non_local(img_noisy, k_nearest=k_nearest, patch_size=patch_size,
                                                      connect_middle=connect_middle, no_overlap=no_overlap)
    else:
        e2v_in, e2v_out = graphify_image(img_noisy, hyperedges=hyper_local, ForwardOH=ForwardOH, BackwardOH=BackwardOH)
        W = None
    # e2v_in, e2v_out = jnp.array(e2v_in), jnp.array(e2v_out)
    img_noisy_cartesian = jnp.moveaxis(jnp.array([np.cos(img_noisy), np.sin(img_noisy)]), 0, -1)
    OH = OrientedHypergraphTuple_e2v_struct(N_pixels, e2v_in, e2v_out, edge_weights=W,
                                            node_features=img_noisy_cartesian.reshape(N_pixels, 2))
    print("S1 image graph generated")
    if return_image:
        return OH, img, img_noisy
    else:
        return OH


if __name__ == "__main__":
    img, img_noisy = artificialIn_SAR_image(256)
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, 6)))
    plt.imshow(img_noisy, cmap="twilight")
    plt.show()
    plt.imshow(img, cmap="hsv")
    plt.show()
    pass