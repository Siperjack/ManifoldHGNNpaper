import pickle

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from generation_phase_image import phase_image_OH
from morphomatics.manifold import Sphere

from OrientedHypergraphs.objects import OHGraphTupleReduced
from OrientedHypergraphs.operators import (
    # FLaplace,
    # OH_Laplace_graph,
    PLaplace,
    regularized_variation,
)

M = Sphere(point_shape=(2,))


def update_regularized_flow(
    OH: OHGraphTupleReduced,
    laplace,
    step_length=1.0,
    lambda_scalar=1.0,
    return_lap=False,
    edge_normalize=True,
    deg_normalize=False,
) -> tuple[OHGraphTupleReduced, jnp.array]:
    """
    The forward Euler scheme for the regularized variation flow.
    :param OH: Named tuple containing the Oriented hypergraph
    :param laplace: Hypergraph Laplacian to use
    :param Man, edge_normalize, deg_normalize: Parameters of the Laplacian, see there.
    :return: The subsequent step in the forward Euler scheme.
    """

    def multi_connec_exp(P, V):
        return jax.vmap(lambda p, v: M.connec.exp(p, v))(P, V)

    step = regularized_variation(
        img_noisy.reshape(len(OH.X), 2),
        OH=OH,
        laplace=laplace,
        Man=M,
        edge_normalize=edge_normalize,
        deg_normalize=deg_normalize,
        lambda_scalar=lambda_scalar,
    )
    OH = OH._replace(X=multi_connec_exp(OH.X, -step_length * step))
    if return_lap:
        lap = laplace(
            OH=OH, Man=M, edge_normalize=edge_normalize, deg_normalize=deg_normalize
        )
        return OH, lap - step, lap
    else:
        return OH, step


if __name__ == "__main__":
    pts = 256
    generate = True
    save_graph = True

    #   Nonlocal denoising parameters
    non_local = False
    k_nearest = 6
    patch_size = 2
    no_overlap = True
    connect_middle = True

    #   Local denoising parameters
    hyper_local = True
    ForwardOH = True
    BackwardOH = False

    step_length = 10 ** (6 * -0.5)
    fidelity_strength = 10 ** (-10 / 4)
    print(
        f"Generating {pts}x{pts} image \n"
        f"save graph = {save_graph}\n"
        f"non_local = {non_local} \n"
        f"k_nearest = {k_nearest} \n"
        f"patch_size = {patch_size} \n"
        f"connect_middle = {connect_middle} \n"
        f"hyper_local = {hyper_local} \n"
        f"ForwardOH = {ForwardOH}\n"
        f"BackwardOH = {BackwardOH}\n"
        f"no_overlap = {no_overlap} \n"
        f"step length = {step_length} \n"
        f"fidelity strength = {fidelity_strength}\n"
    )

    if non_local:
        if connect_middle:
            if no_overlap:
                file_name = (
                    f"phase_image_{pts}_non_local_{k_nearest}_{patch_size}_graph"
                )
            else:
                file_name = f"phase_image_{pts}_non_local_{k_nearest}_{patch_size}_graph_patch_overlap"
        else:
            if no_overlap:
                file_name = f"phase_image_{pts}_non_local_{k_nearest}_{patch_size}"
            else:
                file_name = f"phase_image_{pts}_non_local_{k_nearest}_{patch_size}_patch_overlap"
    else:
        if hyper_local:
            if ForwardOH and BackwardOH:
                file_name = f"phase_image_{pts}_hyper_local"
            elif ForwardOH:
                file_name = f"phase_image_{pts}_hyper_local_ForwardOH"
            elif BackwardOH:
                file_name = f"phase_image_{pts}_hyper_local_BackwardOH"
            else:
                print("invalid parameters for graph img generation")

        else:
            file_name = f"phase_image_{pts}_local"
    if generate:
        OH, img, img_noisy = phase_image_OH(
            pts,
            return_image=True,
            sigma=0.3,
            non_local=non_local,
            hyper_local=hyper_local,
            k_nearest=k_nearest,
            patch_size=patch_size,
            connect_middle=connect_middle,
            no_overlap=no_overlap,
            ForwardOH=ForwardOH,
            BackwardOH=BackwardOH,
        )
        if save_graph:
            file = open(file_name, "wb")
            pickle.dump([OH, img, img_noisy], file)
            file.close()
            print(f"graph image saved as {file_name}")
    else:
        file = open(file_name, "rb")
        OH, img, img_noisy = pickle.load(file)
        file.close()
        print(f"graph image {file_name} loaded")

    img_noisy = jnp.moveaxis(jnp.array([np.cos(img_noisy), np.sin(img_noisy)]), 0, -1)
    OH_F, OH_P, DG = OH, OH, OH
    # DG = OH
    img_x, img_y = len(img[0]), len(img[:, 0])

    end_iter = 15000
    print(f"actual step and fidelity is {step_length} and {fidelity_strength}")
    MSE_best, MSE_best_index = np.infty, 0
    for i in range(end_iter):
        if i == 0:
            print("calculating first Laplace")
        if i % 50 == 0 or i == 5 or i == 15 or i + 1 == end_iter:
            OH_P, regP, lapP = update_regularized_flow(
                OH_P,
                laplace=PLaplace,
                step_length=step_length,
                return_lap=True,
                lambda_scalar=fidelity_strength,
                deg_normalize=True,
                edge_normalize=False,
            )
            print(
                f"Norm of reg is {np.linalg.norm(regP, axis=-1).sum()}, lap is {np.linalg.norm(lapP, axis=-1).sum()}"
                f", and mean is {np.linalg.norm(lapP - regP, axis=-1).sum()/(img_x*img_y)}"
            )
            denoised_im = np.arctan2(OH_P.X[:, 1], OH_P.X[:, 0]).reshape((img_x, img_y))
            plt.imshow(denoised_im, cmap="hsv")
            plt.show()
            MSE = jax.vmap(M.metric.squared_dist)(
                OH_P.X,
                jnp.moveaxis(
                    jnp.array([np.cos(img.flatten()), np.sin(img.flatten())]), 0, -1
                ),
            ) / len(OH_P.X)

            print(f"MSE is {MSE.sum()}:.4f on iteration {i}/{end_iter}")
            if MSE.sum() < MSE_best:
                MSE_best, MSE_best_index = MSE.sum(), i
        else:
            OH_P, step = update_regularized_flow(
                OH_P,
                laplace=PLaplace,
                step_length=step_length,
                lambda_scalar=fidelity_strength,
                deg_normalize=True,
                edge_normalize=False,
            )
        # print("updated flow")
    print(f"Final MSE is {MSE.sum():.4f}")
    print(f"Best MSE is {MSE_best:.4f}, obtained in iteration {MSE_best_index}")

    #   Parameter search for denoising
    # step_lengths = np.logspace(-5, -1, num=10)
    # fidelity_strengths = np.logspace(-5, -1, num=10)
    # MSE_best_list, MSE_best_index_list, param_list = [], [], []
    # for step_length in step_lengths:
    #     for fidelity_strength in fidelity_strengths:
    #         MSE, MSE_best_index = np.infty, 0
    #         OH_P = OH
    #         for i in range(1000):
    #             OH_P, regP, lapP = update_regularized_flow(OH_P, laplace=PLaplace, step_length=step_length,
    #                                                        lambda_scalar=fidelity_strength,
    #                                                        return_lap=True, deg_normalize=True, edge_normalize=True)
    #             MSE_new = jax.vmap(M.metric.squared_dist)(OH_P.X, jnp.moveaxis(
    #                 jnp.array([np.cos(img.flatten()), np.sin(img.flatten())]), 0, -1)) / len(OH_P.X)
    #             if (jnp.isnan(lapP)).any():
    #                 break
    #             if MSE_new.sum() < MSE:
    #                 MSE_best, MSE_best_index = MSE_new.sum(), i
    #
    #         MSE_best_list.append(MSE_best)
    #         MSE_best_index_list.append(MSE_best_index)
    #         param_list.append([step_length, fidelity_strength])
    #
    #         print(f"{len(param_list)}/{len(step_lengths)*len(fidelity_strengths)} of parameterspace searched")
    # file = open(f"Parameter_search_forward", 'wb')
    # pickle.dump([MSE_best_list, MSE_best_index_list, param_list], file)
    # file.close()
