# import os
# import sys

import pickle

import jax
import jax.numpy as jnp
import pyvista as pv
from morphomatics.manifold import Sphere

from OrientedHypergraphs.objects import (
    OHGraphTupleReduced,
    # OrientedHypergraph,
    # OrientedHypergraphTuple_e2v_struct,
    clique_expand,
    # flatten_list_of_arrays,
    # generate_random_signed_incidence_Forward,
    # jraph_to_OH,
)
from OrientedHypergraphs.operators import FLaplace, OH_Laplace, PLaplace
from OrientedHypergraphs.utils import draw_S2_valued_OH  # , animate_S2_valued_OH
from toy_examples.flow_sphere.example_HGs import (
    # OH_rectangle,
    # irregular_graph,
    random_S2_OH,
    # rectangle,
    # tetrahedron,
)

pv.OFF_SCREEN = True  # If not running in a container, set to False
jax.numpy.set_printoptions(precision=2)

# num_node, num_edge = 5, 4
num_node, num_edge = 15, 10
# num_node, num_edge = 10, 15
# num_node, num_edge = 10, 30


M = Sphere(point_shape=(3,))
"""Some examples of (hyper)graphs on S2."""
# OH = jraph_to_OH(tetrahedron(homogeneous_edge_weights=True))
# OH = jraph_to_OH(rectangle(homogeneous_edge_weights=True))
# OH = jraph_to_OH(irregular_graph(homogeneous_edge_weights=True))
# OH = OH_rectangle(homogeneous_edge_weights=True)
OH = random_S2_OH(
    num_node,
    num_edge,
    same_oct=True,
    forward=False,
    backward=False,
    sym=True,
    min_tot_edge_degree=2,
    naive_normalize_deg=False,
)
DG = clique_expand(OH)

# draw_S2_valued_OH(OH)

# draw_OH_as_dhg(DG)
# draw_S2_valued_OH(OH)

# OH_obj = OH
OH = OH.OHGraphTupleReduced
DG = DG.OHGraphTupleReduced

OH_F, OH_P = OH, OH
X_t = [OH.X]
X_F_t = [OH_F.X]
X_P_t = [OH_P.X]
X_click_t = [DG.X]

if (OH.D_in.clip(1e-9) != OH.D_in).all():
    print("Warning: in degree contains 0 and will be clipped in degree normalization")
if (OH.D_out.clip(1e-9) != OH.D_out).all():
    print("Warning: out degree contains 0")


def update_laplace_flow(
    OH: OHGraphTupleReduced,
    laplace,
    step_length=1.0,
    edge_normalize=False,
    deg_normalize=False,
) -> tuple[OHGraphTupleReduced, jnp.array]:
    """
    The forward Euler scheme for the Laplacian flow.
    :param OH: Named tuple containing the Oriented hypergraph
    :param laplace: Hypergraph Laplacian to use
    :param Man, edge_normalize, deg_normalize: Parameters of the Laplacian, see there.
    :return: The subsequent step in the forward Euler scheme.
    """

    def multi_connec_exp(P, V):
        return jax.vmap(lambda p, v: M.connec.exp(p, v))(P, V)

    lap = OH_Laplace(
        OH,
        laplace=laplace,
        Man=M,
        edge_normalize=edge_normalize,
        deg_normalize=deg_normalize,
    )
    OH = OH._replace(X=multi_connec_exp(OH.X, -step_length * lap))
    return OH, lap


def laplace_print(laplace_eval, laplace_str):

    print(f"{laplace_str}-Laplace")
    print(f"0 -> {laplace_eval[0]}")
    print(f"1 -> {laplace_eval[1]}")
    print(f"max norm -> {jnp.sqrt((laplace_eval**2).max())} \n")


max_iter = 1000
for i in range(max_iter):

    OH_F, lapF = update_laplace_flow(
        OH_F, FLaplace, step_length=1, deg_normalize=True, edge_normalize=True
    )
    OH_P, lapP = update_laplace_flow(
        OH_P, PLaplace, step_length=1, deg_normalize=True, edge_normalize=True
    )
    DG, lapg = update_laplace_flow(
        DG, PLaplace, step_length=0.1, deg_normalize=True, edge_normalize=True
    )

    if i % 50 == 10:
        laplace_print(lapF, "F")
        laplace_print(lapP, "P")
        laplace_print(lapg, "g")
    if i % int(max_iter / 1000 + 1) == 0:
        X_F_t.append(OH_F.X)
        X_P_t.append(OH_P.X)
        X_click_t.append(DG.X)

    if (
        jnp.sqrt((lapF**2).max()) < 1e-4
        and jnp.sqrt((lapP**2).max()) < 1e-4
        and jnp.sqrt((lapg**2).max()) < 1e-4
    ):
        print(f"All laplacians converged at {i} iterations")
        break


file_name = "./gifs/laplacian_timeseries"
with open(file_name, "wb") as file:
    pickle.dump([X_F_t, X_P_t, X_click_t], file)
print(f"S2 timeseries saved as {file_name}")


# animate_S2_valued_OH(X_F_t, gif_name="./gifs/Flow_F.gif")
# animate_S2_valued_OH(X_P_t, gif_name="./gifs/Flow_P.gif")
# animate_S2_valued_OH(X_click_t, gif_name="./gifs/Flow_C.gif")

print("Drawing graphs")
draw_S2_valued_OH(OH, file_name="./visualizations/OHinitial.pdf")
draw_S2_valued_OH(OH_F, file_name="./visualizations/OHfinalF.pdf")
draw_S2_valued_OH(OH_P, file_name="./visualizations/OHfinalP.pdf")
draw_S2_valued_OH(DG, file_name="./visualizations/OHfinalg.pdf")
laplace_print(FLaplace(OH_F, M, edge_normalize=True, deg_normalize=True), "F")
laplace_print(PLaplace(OH_P, M, edge_normalize=True, deg_normalize=True), "P")
laplace_print(PLaplace(DG, M, edge_normalize=True, deg_normalize=True), "g")  # g is the Laplacian of the clique expansion and PLaplace=FLaplace

if __name__ == "__main__":
    pass
