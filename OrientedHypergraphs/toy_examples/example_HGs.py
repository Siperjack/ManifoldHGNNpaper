import jax.numpy as jnp
import jraph
import numpy as np
from OrientedHypergraphs.objects import OrientedHypergraph, OrientedHypergraph_e2v_struct, generate_random_signed_incidence, \
    OrientedHypergraphTuple_e2v_struct, orient_hypergraph_one_to_all, \
    generate_random_signed_incidence_Forward, H_to_e2v, e2v_to_H



def OH_rectangle(homogeneous_edge_weights=True) -> OrientedHypergraph:
    """Define a four node graph, each node has a feature on the 2-sphere."""

    node_features = jnp.array(
        [[0., 0., 1.], [0., 1., 0.], [1 / jnp.sqrt(3), 1 / jnp.sqrt(3), -1 / jnp.sqrt(3)], [1., 0., 0.]])

    # senders = jnp.array([0, 1, 2, 3])
    # receivers = jnp.array([1, 2, 3, 0])

    # senders = [[0], [1], [2], [3], [3], [1], [2], [0]]
    # receivers = [[1], [2], [3], [0], [2], [0], [0], [3]]

    senders = [[0], [1], [2], [3], [3], [1, 2], [0]]
    receivers = [[1, 2], [3], [0], [2], [0], [0], [3]]

    n_node = len(node_features)
    n_edge = len(senders)

    if homogeneous_edge_weights:
        edges = jnp.array([[.25] * int(n_edge)]).T
    else:
        edges = jnp.array([[.125], [.125], [.15], [.6]])
    OH = OrientedHypergraph_e2v_struct(n_node, senders, receivers, node_features=node_features, edge_weights=edges)
    return OH


def OH_coinciding_mean(homogeneous_edge_weights=True) -> OrientedHypergraph:
    """Define a four node graph, each node has a feature on the 2-sphere."""

    node_features = jnp.array(
        [[0., 0., 1.], [0., 1., 0.], [1 / jnp.sqrt(3), 1 / jnp.sqrt(3), -1 / jnp.sqrt(3)], [1., 0., 0.]])

    # senders = jnp.array([0, 1, 2, 3])
    # receivers = jnp.array([1, 2, 3, 0])

    # senders = [[0], [1], [2], [3], [3], [1], [2], [0]]
    # receivers = [[1], [2], [3], [0], [2], [0], [0], [3]]

    senders = [[0], [1], [2], [3], [3], [1, 2], [0]]
    receivers = [[1, 2], [3], [0], [2], [0], [0], [3]]

    n_node = len(node_features)
    n_edge = len(senders)

    if homogeneous_edge_weights:
        edges = jnp.array([[.25] * int(n_edge)]).T
    else:
        edges = jnp.array([[.125], [.125], [.15], [.6]])
    OH = OrientedHypergraph_e2v_struct(n_node, senders, receivers, node_features=node_features, edge_weights=edges)
    return OH


def rectangle(homogeneous_edge_weights=True) -> jraph.GraphsTuple:
    """Define a four node graph, each node has a feature on the 2-sphere."""

    node_features = jnp.array(
        [[0., 0., 1.], [0., 1., 0.], [1 / jnp.sqrt(3), 1 / jnp.sqrt(3), -1 / jnp.sqrt(3)], [1., 0., 0.]])

    # senders = jnp.array([0, 1, 2, 3])
    # receivers = jnp.array([1, 2, 3, 0])

    senders = jnp.array([0, 1, 2, 3, 3, 2, 1, 0])
    receivers = jnp.array([1, 2, 3, 0, 2, 1, 0, 3])

    n_node = jnp.array([len(node_features)])
    n_edge = jnp.array([len(senders)])

    if homogeneous_edge_weights:
        edges = jnp.array([[.25] * int(n_edge[0])]).T
    else:
        edges = jnp.array([[.125], [.125], [.15], [.6], [.15], [.125], [.125], [.6]])

    global_context = None
    graph = jraph.GraphsTuple(
        nodes=node_features,
        edges=edges,
        senders=senders,
        receivers=receivers,
        n_node=n_node,
        n_edge=n_edge,
        globals=global_context
    )
    return graph


def irregular_graph(homogeneous_edge_weights=False) -> jraph.GraphsTuple:
    """Define a four node graph, each node has a feature on the 2-sphere."""

    node_features = jnp.array([[0., 0., 1.], [0., 1., 0.], [0., 1 / jnp.sqrt(2), 1 / jnp.sqrt(2)], [1., 0., 0.]])

    senders = jnp.array([0, 2, 3, 2, 1, 0, 0, 3])
    receivers = jnp.array([1, 0, 0, 3, 0, 2, 3, 2])

    if homogeneous_edge_weights:
        edges = jnp.array([[.25], [.25], [.25], [.25], [.25], [.25], [.25], [.25]])
    else:
        edges = jnp.array([[.125], [.125], [.15], [.6], [.125], [.125], [.15], [.6]])

    n_node = jnp.array([len(node_features)])
    n_edge = jnp.array([len(senders)])

    global_context = None
    graph = jraph.GraphsTuple(
        nodes=node_features,
        edges=edges,
        senders=senders,
        receivers=receivers,
        n_node=n_node,
        n_edge=n_edge,
        globals=global_context
    )
    return graph


def tetrahedron(homogeneous_edge_weights=True) -> jraph.GraphsTuple:
    """Define a tetrahedron graph on the 2-sphere."""

    node_features = jnp.array([[0., 0., 1.], [jnp.sqrt(8 / 9), 0., -1 / 3], [-jnp.sqrt(2 / 9), jnp.sqrt(2 / 3), -1 / 3],
                               [-jnp.sqrt(2 / 9), -jnp.sqrt(2 / 3), -1 / 3]])

    # senders = jnp.array([0, 1, 2, 3])
    # receivers = jnp.array([1, 2, 3, 0])

    senders = jnp.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    receivers = jnp.array([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2])

    n_node = jnp.array([len(node_features)])
    n_edge = jnp.array([len(senders)])

    if homogeneous_edge_weights:
        edges = jnp.atleast_2d(jnp.repeat(1 / n_node, int(n_edge[0]))).T
    else:
        edges = jnp.array([[.125], [.125], [.15], [.6], [.15], [.15], [.15], [.15], [.15], [.15], [.15], [.15]])

    global_context = None
    graph = jraph.GraphsTuple(
        nodes=node_features,
        edges=edges,
        senders=senders,
        receivers=receivers,
        n_node=n_node,
        n_edge=n_edge,
        globals=global_context
    )
    return graph


def random_feature(n_node, gaussian=False, sd=1):
    if gaussian:   # Better distribution around a point on the sphere would be sampling in tangent space and use exp
        random_R3 = np.random.normal(loc=jnp.ones((n_node, 3)), size=(n_node, 3), scale=sd)
    else:
        random_R3 = np.random.uniform(low=-1, high=1, size=(n_node, 3))
    return random_R3 / np.linalg.norm(random_R3, axis=1).reshape(n_node, 1)


def random_S2_OH(N, M, naive_normalize_deg=False, same_oct=False, scale=1, forward=False, backward=False, sym=True,
                 min_tot_edge_degree=2):
    """
    Makes a OrientedHypergraph class object embedded on the sphere according to the parameters
    :param N: #nodes
    :param M: #edges
    :param naive_normalize_edge: normalize by the constant max degree for stability
    :param same_oct:Restrict the S2 embeddings to an octant
    :param scale: If sampled from Gausian around a mean, this is the variance of the Gaussian
    :param forward: Make defining incidence matrix have only one -1 in each column. With sym: one-to-all an unoriented H
    :param backward: Swaps all signed entries in the incidence matrix of a forward hypergraph
    :param sym: Make incidence matrix the union of H and H^T. With forward: one-to-all an unoriented H
    :param min_tot_edge_degree: For forward hypergraphs to ensure symmetric Laplacians makes sense (up to a factor of 2)
    :return: OrientedHypergraph class object defined by the incidence matrix and embeddings randomly generated
    """
    random_R3 = jnp.array(random_feature(N, sd=scale))
    if same_oct:
        random_R3 = jnp.abs(random_R3)
    random_H = generate_random_signed_incidence(N, M, sym=sym, min_tot_edge_degree=min_tot_edge_degree)
    if forward:
        if sym:
            e2v, _ = H_to_e2v(np.abs(random_H))
            e2v_in, e2v_out = orient_hypergraph_one_to_all(e2v)
            random_H = e2v_to_H(N, e2v_in, e2v_out)
        else:
            random_H = generate_random_signed_incidence_Forward(N, M, min_tot_edge_degree=min_tot_edge_degree)
        if backward:
            random_H = random_H*-1
    OH = OrientedHypergraph(random_H, node_features=random_R3)
    if naive_normalize_deg:
        OH.W = OH.W / OH.max_deg_in
        return OrientedHypergraph(random_H, node_features=random_R3, edge_weights=OH.W)
    return OH



def dhg_converter(dhg_data):
    num_classes = dhg_data.num_classes
    num_nodes = dhg_data.num_vertices
    edge_list = dhg_data.edge_list
    e2v_in, e2v_out = orient_hypergraph_one_to_all(edge_list)
    X = dhg_data.features
    OH = OrientedHypergraphTuple_e2v_struct(num_nodes=num_nodes, e2v_in=e2v_in, e2v_out=e2v_out,
                                            node_features=X)
    return OH, num_classes


# if __name__ == '__main__':
#     N = 5
#     M = 10
#     data = dhg_converter(CocitationCora)

# def random_graph_sphere(key, n_node=5, n_edge=None, mean=jnp.array([0, 0, 1]), covariance_matrix=jnp.eye(2),
#                         n_channel=1):
#     """Random graph on the 2D-sphere. The vertices are sampled from a normal distribution in the tangent space at the
#     North Pole and mapped down with the exponential map.
#
#     @param key: PRNGKey
#     @param n_node: number of nodes
#     @param n_edge: number of edges (n_edge <= n_node * (n_node - 1)); if None, then the complete graph is returned
#     @param mean: mean around which the vertices are sampled
#     @param covariance_matrix: 2x2 SPD matrix modeling the covariance of the normal distribution
#     @param n_channel: number of channels
#     @return: graph object with vertex features on S2
#     """
#     assert n_edge is None or n_edge <= n_node * (n_node - 1)
#
#     S = Sphere()
#     SO = SO3()
#     north_pole = jnp.array([0, 0, 1])
#
#     # create rotaton matrix if mean is not North Pole
#     if jnp.array_equal(mean, north_pole):
#         R = jnp.eye(3)
#     elif jnp.array_equal(mean, jnp.array([0, 0, -1])):
#         # 180-degree rotation about x-axis
#         R = -jnp.eye(3)
#         R = R.at[0, 0].set(1)
#     else:
#         # unit rotation axis
#         k = jnp.cross(north_pole, mean)
#         k = k / jnp.linalg.norm(k)
#
#         K = jnp.zeros((1, 3, 3))
#         K = K.at[0, 0, 1].set(-k[2])
#         K = K.at[0, 1, 0].set(k[2])
#         K = K.at[0, 0, 2].set(k[1])
#         K = K.at[0, 2, 0].set(-k[1])
#         K = K.at[0, 1, 2].set(-k[0])
#         K = K.at[0, 2, 1].set(k[0])
#         theta = jnp.arccos(jnp.dot(mean, north_pole))
#
#         R = SO.connec.exp(theta * K)[0]
#
#     # create nodes around the North Pole for each channel
#     # random vectors
#     node_feature_vectors = jax.random.multivariate_normal(key, jnp.zeros((2,)), covariance_matrix, shape=[n_node])
#     channels = []
#     for _ in range(n_channel):
#         features = []
#         for _vector in node_feature_vectors:
#             vector = jnp.zeros((3,))
#             # tangent vector at mean
#             vector = vector.at[:2].set(_vector)
#             # shoot and rotate
#             features.append(R @ S.connec.exp(north_pole, vector))
#
#         channels.append(jnp.array(features))
#
#     node_features = jnp.swapaxes(jnp.array(channels), 0, 1)  # node before channel dimension
#
#     # create edges
#     if n_edge is None or n_edge == n_node * (n_node - 1):
#         # complete graph
#         n_edge = n_node * (n_node - 1)
#
#         senders = []
#         receivers = []
#         for i in range(n_node):
#             for j in range(n_node):
#                 if i != j:
#                     senders.append(i)
#                     receivers.append(j)
#
#         senders = jnp.array(senders)
#         receivers = jnp.array(receivers)
#
#     edges = jnp.atleast_2d(jnp.repeat(1, n_edge)).T
#
#     # create graph
#     global_context = None
#     graph = jraph.GraphsTuple(
#       nodes=node_features,
#       edges=edges,
#       senders=jnp.array(senders),
#       receivers=jnp.array(receivers),
#       n_node=jnp.array([n_node]),
#       n_edge=jnp.array([n_edge]),
#       globals=global_context
#       )
#
#     return graph


# def multi_tetrahedron_graph(homogeneous_edge_weights=True) -> jraph.GraphsTuple:
#     node_features_a = jnp.array([[0., 0., 1.], [jnp.sqrt(8/9), 0., -1/3], [-jnp.sqrt(2/9), jnp.sqrt(2/3), -1/3],
#                                [-jnp.sqrt(2/9), -jnp.sqrt(2/3), -1/3]])
#
#     node_features_b = - node_features_a
#
#     node_features = jnp.stack((node_features_a, node_features_b))
#
#     # senders = jnp.array([0, 1, 2, 3])
#     # receivers = jnp.array([1, 2, 3, 0])
#
#     senders = jnp.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
#     receivers = jnp.array([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2])
#
#     n_node = jnp.array([len(node_features_a)])
#     n_edge = jnp.array([len(senders)])
#     if homogeneous_edge_weights:
#         edges = jnp.atleast_2d(jnp.repeat(1/n_node, int(n_edge[0]))).T
#     else:
#         edges = jnp.array([[.125], [.125], [.15], [.6], [.15], [.15], [.15], [.15], [.15], [.15], [.15], [.15]])
#
#     global_context = None
#     graph = jraph.GraphsTuple(
#       nodes=node_features,
#       edges=edges,
#       senders=senders,
#       receivers=receivers,
#       n_node=n_node,
#       n_edge=n_edge,
#       globals=global_context
#       )
#     return graph
#
#
# def multi_rectangle_graph(homogeneous_edge_weights=True) -> jraph.GraphsTuple:
#     node_features_a = jnp.array([[0., 0., 1.], [0., 1., 0.], [1/jnp.sqrt(3), 1/jnp.sqrt(3), -1/jnp.sqrt(3)],
#                                  [1., 0., 0.]])
#
#     node_features_b = - node_features_a
#
#     node_features = jnp.swapaxes(jnp.stack((node_features_a, node_features_b)), 0, 1)
#
#     # senders = jnp.array([0, 1, 2, 3])
#     # receivers = jnp.array([1, 2, 3, 0])
#
#     senders = jnp.array([0, 1, 2, 3, 3, 2, 1, 0])
#     receivers = jnp.array([1, 2, 3, 0, 2, 1, 0, 3])
#
#     n_node = jnp.array([len(node_features_a)])
#     n_edge = jnp.array([len(senders)])
#     if homogeneous_edge_weights:
#         edges = jnp.array([[.25], [.25], [.25], [.25], [.25], [.25], [.25], [.25]])
#     else:
#         edges = jnp.array([[.125], [.125], [.15], [.6], [.125], [.125], [.15], [.6]])
#
#     global_context = None
#     graph = jraph.GraphsTuple(
#       nodes=node_features,
#       edges=edges,
#       senders=senders,
#       receivers=receivers,
#       n_node=n_node,
#       n_edge=n_edge,
#       globals=global_context
#       )
#     return graph
#