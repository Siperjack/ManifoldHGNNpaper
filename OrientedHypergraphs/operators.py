import jax
import jax.numpy as jnp
# import jraph
import numpy as np
# from jax import tree_util as tree
# from jax.experimental import sparse
from morphomatics.graph.operators import mfdg_laplace
from morphomatics.manifold import Manifold
# from morphomatics.nn.wFM_layers import wFM
from morphomatics.stats import ExponentialBarycenter as FrechetMean

from OrientedHypergraphs.objects import (
    OHGraphTupleReduced,
    OHtuple_to_jraph_edgeless,
    # OrientedHypergraph,
    # calc_e2v2v_bool,
    # flatten_list_of_arrays,
)


def grad_F_unweighted(
    OH: OHGraphTupleReduced, Man: Manifold
) -> tuple[jnp.array, jnp.array]:
    """
    Compute the unweighted Frechet-mean gradient of Stokke Thesis 2024
    :param OH: Named tuple containing the Oriented hypergraph
    :param Man: A connected and complete Riemannian manifold with well-defined Fréchet mean
    :return: tangent vectors in the discrete tangent bundle and corresponding tangent space base points
    """
    FrechetE_in, FrechetE_out = np.zeros(shape=(2, OH.num_edge, len(OH.X[0])))
    for i in range(
        OH.num_edge
    ):  # TODO: Make e_in_deg constant. One way is through introducing weights in Fréchet
        FrechetE_in[i] = FrechetMean().compute(Man, OH.X[np.array(OH.e2v_in[i])])
        FrechetE_out[i] = FrechetMean().compute(Man, OH.X[np.array(OH.e2v_out[i])])
    LogE_vec = jax.vmap(Man.connec.log)(FrechetE_in, FrechetE_out)
    return LogE_vec, FrechetE_in


def div_F_sym_weighted_squared(
    OH: OHGraphTupleReduced,
    H_f: jnp.array,
    base_points,
    Man: Manifold,
    edge_normalize=False,
    deg_normalize=False,
):
    """
    Compute the Fréchet divergence from Stokke Thesis
    :param OH: Named tuple containing the Oriented hypergraph
    :param Man: A connected and complete Riemannian manifold
    :param H_f: list of tangent vectors corresponding to edges
    :param base_points: list of tangent vectors base points corresponding to H_f
    :param edge_normalize: True-> gamma=0, False->gamma=1 from Stokke Thesis
    :param deg_normalize: True-> eta=1, False->eta=0 from Stokke Thesis
    :return: list of Laplace tangent vectors for each node
    """
    if edge_normalize:
        W = OH.W / OH.E_in
    else:
        W = OH.W
    if deg_normalize:
        D_in = OH.D_in
        D_in += D_in + (D_in == 0)
    else:
        D_in = jnp.ones(shape=np.shape(OH.D_in))

    H_f_weighted = H_f * W[..., None]

    # PT_start = jnp.array([item for item in jnp.arange(OH.num_edge) for _ in range(len(OH.e2v_in[item]))])
    # PT_stop = flatten_list_of_arrays(OH.e2v_in)
    PT_start, PT_stop = OH.PT_num_F[0], OH.PT_num_F[1]

    PTedH_f = jax.vmap(Man.connec.transp)(
        base_points[PT_start], OH.X[PT_stop], H_f_weighted[PT_start]
    )
    div_unormalized = jax.ops.segment_sum(PTedH_f, PT_stop, num_segments=OH.num_node)
    div = div_unormalized / D_in[..., None]
    return div


def FLaplace(
    OH: OHGraphTupleReduced, Man: Manifold, edge_normalize=False, deg_normalize=False
):
    """
    Compute the Fréchet 2-Laplacian of Stokke Thesis 2024
    :param OH: Named tuple containing the Oriented hypergraph
    :param Man: A connected and complete Riemannian manifold with well-defined Fréchet mean
    :param edge_normalize: True-> gamma=0, False->gamma=1 from Stokke Thesis
    :param deg_normalize: True-> eta=1, False->eta=0 from Stokke Thesis
    :return: resulting Laplace tangent vectors ordered as nodes
    """
    gradsF, base_points = grad_F_unweighted(OH, Man)
    Laplace = -div_F_sym_weighted_squared(
        OH,
        gradsF,
        base_points,
        Man,
        edge_normalize=edge_normalize,
        deg_normalize=deg_normalize,
    )
    return Laplace


def grad_P_unweighted(OH: OHGraphTupleReduced, Man: Manifold) -> jnp.array:
    """
    Compute the unweighted pairwise gradient of Stokke Thesis 2024 with antipodal points logarithms set to 0.
    :param OH: Named tuple containing the Oriented hypergraph
    :param Man: A connected and complete Riemannian manifold
    :return: tangent vectors in the pairwise discrete tangent bundle and corresponding tangent space base points
    """
    indexes_logs_edges_out, indexes_logs_nodes_in, indexes_logs_nodes_out = OH.PT_num_P[
        0:3
    ]

    # Trows away antipodal points that may arise in big images as a safeguard to NaNs. Based on the
    # assumption that antipodal points, if they arise at all, are sufficiently sparse in the pairwise tangent matrices
    antipodal_pairs = (
        jnp.linalg.norm(
            (OH.X[indexes_logs_nodes_in] + OH.X[indexes_logs_nodes_out]), axis=-1
        )
        > 1e-3
    )[..., None]
    Logs = jax.vmap(Man.connec.log)(
        OH.X[indexes_logs_nodes_in] * antipodal_pairs,
        OH.X[indexes_logs_nodes_out] * antipodal_pairs,
    )
    if (jnp.linalg.norm(Logs, axis=-1) > 10).any():
        print("to big log found")
    if jnp.isnan(Logs).any():
        print("error in antipodal-corrected Logs found")
    if jnp.isnan(
        jax.vmap(Man.connec.log)(
            OH.X[indexes_logs_nodes_in], OH.X[indexes_logs_nodes_out]
        )
    ).any():
        print("error in Logs found")
    # Logs = jax.ops.segment_sum(Logs, indexes_logs_edges_out, num_segments=len(OH.PT_num_P[6]))
    return Logs


def div_P_sym_weighted_squared(
    OH: OHGraphTupleReduced,
    H_f: jnp.array,
    Man: Manifold,
    edge_normalize=False,
    deg_normalize=False,
    pairwise_summed=False,
):
    """
    Compute the pairwise divergence from Stokke Thesis with antipodal points logarithms set to 0.
    :param OH: Named tuple containing the Oriented hypergraph
    :param Man: A connected and complete Riemannian manifold
    :param H_f: flattened list of tangent vectors corresponding to ordering indexes_logs_nodes_in in PT_num_P
    :param edge_normalize: True-> gamma=0, False->gamma=1 from Stokke Thesis
    :param deg_normalize: True-> eta=1, False->eta=0 from Stokke Thesis
    :param pairwise_summed: True-> epsilon=1, False->epsilon=0 from Stokke Thesis
    :return: list of Laplace tangent vectors for each node
    """
    W = OH.W
    if not pairwise_summed:
        W = OH.W / (OH.E_in * OH.E_out)

    if edge_normalize:
        W = W / OH.E_in

    if deg_normalize:
        D_in = OH.D_in
        D_in += D_in + (D_in == 0)
    else:
        D_in = jnp.ones(shape=np.shape(OH.D_in))

    H_f = jax.ops.segment_sum(H_f, OH.PT_num_P[0], num_segments=len(OH.PT_num_P[6]))
    (
        indexes_PT_edges_in,
        indexes_PT_nodes_in,
        indexes_PT_nodes_out,
        indexes_P_edges_in_div,
        indexes_P_edges_segsum,
    ) = OH.PT_num_P[3:]
    antipodal_pairs = (
        jnp.linalg.norm(
            (OH.X[indexes_PT_nodes_out] + OH.X[indexes_PT_nodes_in]), axis=-1
        )
        > 1e-3
    )[..., None]
    # antipodal_pairs = 1
    PTedH_f = jax.vmap(Man.connec.transp)(
        OH.X[indexes_PT_nodes_in] * antipodal_pairs,
        OH.X[indexes_PT_nodes_out] * antipodal_pairs,
        H_f[indexes_PT_edges_in],
    )

    PTedH_f_summed = jax.ops.segment_sum(
        PTedH_f, indexes_P_edges_segsum, num_segments=len(OH.PT_num_P[6])
    )
    PTedH_f_summed_weighted = PTedH_f_summed * W[indexes_P_edges_in_div][..., None]
    div_unormalized = jax.ops.segment_sum(
        PTedH_f_summed_weighted, OH.PT_num_F[-1], num_segments=(len(OH.X))
    )
    div = div_unormalized / D_in[..., None]
    return div


def PLaplace(
    OH: OHGraphTupleReduced, Man: Manifold, edge_normalize=False, deg_normalize=False
):
    """
    Compute the pairwise 2-Laplacian from Stokke Thesis
    :param OH: Named tuple containing the Oriented hypergraph
    :param Man: A connected and complete Riemannian manifold
    :param edge_normalize: True-> gamma=0, False->gamma=1 from Stokke Thesis
    :param deg_normalize: True-> eta=1, False->eta=0 from Stokke Thesis
    :param pairwise_summed: True-> epsilon=1, False->epsilon=0 from Stokke Thesis
    :return: resulting Laplace tangent vectors ordered as nodes
    """
    gradsP = grad_P_unweighted(OH, Man)
    Laplace = -div_P_sym_weighted_squared(
        OH, gradsP, Man, edge_normalize=edge_normalize, deg_normalize=deg_normalize
    )
    return Laplace


def OH_Laplace(
    OH: OHGraphTupleReduced,
    Man: Manifold,
    laplace,
    edge_normalize=False,
    deg_normalize=False,
):
    """
    Wrapper function for the hypergraph Laplacians
    """
    return laplace(OH, Man, edge_normalize, deg_normalize)


def OH_Laplace_graph(
    OH: OHGraphTupleReduced, Man: Manifold, edge_normalize=False, deg_normalize=False
):
    """
    Wrapper function for the standard graph Laplacian from morphomatics to be applied on standard graphs represented as
    a OHGraphTuple. Will fail if there are any pure hyperedges.
    """
    return mfdg_laplace(Man, OHtuple_to_jraph_edgeless(OH, keep_edges=True))


def manifold_feature_distance(X: jnp.array, Y: jnp.array, Man: Manifold):
    """
    :param X: Data to fit, f0 in Stokke Thesis data fidelity term
    :param Y: Current features, f in Stokke Thesis data fidelity term
    :param Man: A connected and complete Riemannian manifold
    :return: The data fidelity functional
    """
    d_Ms = jax.vmap(Man.metric.squared_dist)(X, Y)
    return d_Ms.sum()


def manifold_feature_distance_variation(X: jnp.array, Y: jnp.array, Man: Manifold):
    """
    :param X: Data to fit, f0 in Stokke Thesis data fidelity term
    :param Y: Current features, f in Stokke Thesis data fidelity term
    :param Man: A connected and complete Riemannian manifold
    :return: The variation of the data fidelity functional
    """
    var = -jax.vmap(Man.connec.log)(Y, X)
    return var


def regularized_variation(
    data,
    laplace,
    OH: OHGraphTupleReduced,
    Man: Manifold,
    lambda_scalar=1,
    edge_normalize=False,
    deg_normalize=False,
):
    """
    :param data: Data for data fidelty term for fitting
    :param laplace: The laplacian that related to regularization of some energy functional of a gradient
    :param OH: Named tuple containing the Oriented hypergraph
    :param Man: A connected and complete Riemannian manifold
    :param lambda_scalar: Fidelity strength
    :param edge_normalize: True-> gamma=0, False->gamma=1 from Stokke Thesis
    :param deg_normalize: True-> eta=1, False->eta=1 from Stokke Thesis
    :return: regularized_variation in the form of a tangent vector on each node at its feature in the manifold
    """
    laplace_val = laplace(
        OH, Man, edge_normalize=edge_normalize, deg_normalize=deg_normalize
    )
    fidelity_val = lambda_scalar * manifold_feature_distance_variation(data, OH.X, Man)
    return laplace_val + fidelity_val
