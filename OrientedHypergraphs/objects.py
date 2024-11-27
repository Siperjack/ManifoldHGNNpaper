import numpy as np
import jax.numpy as jnp
import jraph
import matplotlib.pyplot as plt

# import dhg as dhg

from typing import Any, NamedTuple, Iterable, Mapping, Union, Optional
np.random.seed(49)

ArrayTree = Union[jnp.ndarray, Iterable['ArrayTree'], Mapping[Any, 'ArrayTree']]

class OHGraphTupleReduced(NamedTuple):
    """Named tuple format of the OrientedHypergraph class without incidence matrix stored"""
    X: Optional[ArrayTree]
    W: Optional[ArrayTree]
    e2v_in: list[jnp.ndarray]
    e2v_out: list[jnp.ndarray]
    v2e_in: list[jnp.ndarray]
    v2e_out: list[jnp.ndarray]
    globals: jnp.ndarray
    num_node: jnp.ndarray
    feature_dim: jnp.ndarray
    num_edge: jnp.ndarray
    D_in: jnp.ndarray
    D_out: jnp.ndarray
    E_in: jnp.ndarray
    E_out: jnp.ndarray
    max_deg_in: jnp.integer
    max_deg_out: jnp.integer
    max_edge_deg_in: jnp.integer
    max_edge_deg_out: jnp.integer
    v2v_bool: jnp.ndarray
    PT_num_F: Optional[jnp.ndarray]
    PT_num_P: Optional[jnp.ndarray]


def OrientedHypergraphTuple_e2v_struct(num_nodes, e2v_in, e2v_out, node_features=None, edge_weights=None,
                                       graph_feature=None):
    """
    Lets you define your oriented hypergraph with edges-to-vertices, which is one-to-one with the signed incidence
    Params similar to OrientedHypergraph class member variables
    """
    if node_features is None:
        node_features = jnp.ones((num_nodes, 1))  # N array of node features on Manifold
    if edge_weights is None:
        print("No weights given, default is 1")
        edge_weights = jnp.ones(len(e2v_in))   # / num_nodes
    else:
        W = jnp.array([edge_weights]).ravel() * jnp.ones(len(e2v_in))
        edge_weights = W.reshape(len(e2v_in))  # M array of hyper edge weights
    print_progress = True
    v2e_in, v2e_out = None, None #e2v_to_v2e(num_nodes, e2v_in, e2v_in)
    if print_progress:
        print("e2v_to_v2e done")
    D_in, D_out = calcDegreeDiag_from_e2v(num_nodes, e2v_in, e2v_out, edge_weights)
    # D_in, D_out = calcDegreeDiag(num_nodes, v2e_in, v2e_out, edge_weights)

    if print_progress:
        print("degree done")
    E_in, E_out = calcEdgeDegreeDiag(len(e2v_in), e2v_in, e2v_out)
    if print_progress:
        print("edge degree done")
    v2v_bool = calc_v2v_bool_sparse(e2v_in, e2v_out)
    if print_progress:
        print("v2v_bool done")
    PT_start_F = jnp.array([item for item in jnp.arange(len(e2v_in)) for _ in range(len(e2v_in[item]))])
    PT_stop_F = flatten_list_of_arrays(e2v_in)
    PT_num_F = jnp.array([PT_start_F, PT_stop_F])
    PT_num_P = calc_PLaplace_indexes_for(e2v_in, e2v_out)
    return OHGraphTupleReduced(X=node_features, W=edge_weights, e2v_in=e2v_in, e2v_out=e2v_out, v2e_in=v2e_in,
                              v2e_out=v2e_out, globals=graph_feature, num_node=num_nodes, feature_dim=len(node_features[0]),
                              num_edge=len(e2v_in), D_in=D_in, D_out=D_out, E_in=E_in, E_out=E_out,
                              max_deg_in=jnp.max(D_in), max_deg_out=jnp.max(D_out), max_edge_deg_in=jnp.max(E_in),
                              max_edge_deg_out=jnp.max(E_out),
                              v2v_bool=v2v_bool, PT_num_F=PT_num_F, PT_num_P=PT_num_P)


class OrientedHypergraph:
    def __init__(self, incidence_matrix, node_features=None, edge_weights=None, graph_feature=None):
        self.H = incidence_matrix  # NxM incidence matrix valued {-1, 0, 1} fully determining the hypergraph topology
        if node_features is None:
            self.X = jnp.ones((len(self.H), 1))  # N array of node features on Manifold
        else:
            self.X = node_features
        if edge_weights is None:
            # print("No weights given, default is 1/N")
            # self.W = jnp.ones(len(self.H[0]))/len(self.X)
            print("No weights given, default is 1")
            self.W = jnp.ones(len(self.H[0]))
        else:
            W = jnp.array([edge_weights]).ravel() * jnp.ones(len(self.H[0]))
            self.W = W.reshape(len(self.H[0]))  # M array of hyper edge weights
        """
        e2v_in: array of the M edges with the indices of their e_in set
        e2v_out: array of the M edges with the indices of their e_out set
        v2e_in: array of the N edges with the indices of their N_in set
        v2e_out: array of the N edges with the indices of their N_out set
        v2v_bool: binary adjacency matrix
        """

        self.e2v_in, self.e2v_out = self.H_to_e2v()
        self.v2e_in, self.v2e_out = self.H_to_v2e()
        self.globals = graph_feature
        self.num_node = len(self.X)
        self.feature_dim = len(self.X[0])
        self.num_edge = len(self.W)
        self.max_deg_in = ((self.H > 0).sum(axis=1).max())
        self.max_deg_out = ((self.H < 0).sum(axis=1).max())
        self.max_edge_deg_in = ((self.H > 0).sum(axis=0).max())
        self.max_edge_deg_out = ((self.H < 0).sum(axis=0).max())
        self.D_in, self.D_out = self.calcDegreeDiag()
        self.E_in, self.E_out = self.calcEdgeSizeDiag()
        self.e2v2v = self.calc_e2v2v_bool()
        self.v2v_bool = self.calc_v2v_bool()
        self.OHGraphTupleReduced = OrientedHypergraphTuple_e2v_struct(num_nodes=self.num_node, e2v_in=self.e2v_in,
                                                                      e2v_out=self.e2v_out,
                                                                      node_features=self.X, edge_weights=self.W)

    def H_to_e2v(self):
        e_list_in, e_list_out = [], []
        for m_edge in self.H.T:
            edge_in, edge_out = [], []
            for n, n_node in enumerate(m_edge):
                if n_node == 1:
                    edge_in.append(n)
                elif n_node == -1:
                    edge_out.append(n)
            assert len(edge_in + edge_out) == len(jnp.unique(jnp.array(edge_in + edge_out))), \
                "an e_in and e_out contain a node in common, this is not allowed in this class"
            e_list_in.append(jnp.array(edge_in))
            e_list_out.append(jnp.array(edge_out))
        return e_list_in, e_list_out

    def H_to_v2e(self):
        v_list_in, v_list_out = [], []
        for n_node in self.H:
            node_in, node_out = [], []
            for m, m_edge in enumerate(n_node):
                if m_edge == 1:
                    node_in.append(m)
                elif m_edge == -1:
                    node_out.append(m)
            v_list_in.append(jnp.array(node_in))
            v_list_out.append(jnp.array(node_out))
        return v_list_in, v_list_out

    def e2v_to_H(self):
        H = jnp.zeros((self.num_node, self.num_edge))
        for m in range(self.num_edge):
            for node_in in self.e2v_in[m]:
                H[node_in, m] = 1
            for node_out in self.e2v_out[m]:
                H[node_out, m] = -1
        return H

    def calcDegreeDiag(self):
        D_in, D_out = np.zeros((2, self.num_node))
        for n in range(self.num_node):
            if len(self.v2e_in[n]) == 0:
                D_in[n] = 0
            else:
                D_in[n] = self.W[self.v2e_in[n]].sum()
            if len(self.v2e_out[n]) == 0:
                D_out[n] = 0
            else:
                D_out[n] = self.W[self.v2e_out[n]].sum()
        return jnp.array(D_in), jnp.array(D_out)

    def calcEdgeSizeDiag(self):
        E_in, E_out = np.zeros((2, self.num_edge))
        for m in range(self.num_edge):
            E_in[m] = len(self.e2v_in[m])
            E_out[m] = len(self.e2v_out[m])
        return jnp.array(E_in), jnp.array(E_out)

    def calc_e2v2v_bool(self):
        triplets = np.zeros((self.num_edge, self.num_node, self.num_node))
        for m in range(self.num_edge):
            for i in self.e2v_in[m]:
                for j in self.e2v_out[m]:
                    triplets[m, i, j] = 1
        return jnp.array(triplets)

    def calc_v2v_bool(self):
        triplets = np.zeros((self.num_node, self.num_node))
        for m in range(self.num_edge):
            for i in self.e2v_in[m]:
                for j in self.e2v_out[m]:
                    triplets[i, j] = 1
        return jnp.array(triplets)

    def _replace(self, **kwargs):
        return OrientedHypergraph(kwargs.get('H', self.H), kwargs.get('X', self.X),
                                  kwargs.get('W', self.W), kwargs.get('globals', self.globals))


def OrientedHypergraph_e2v_struct(num_nodes, e2v_in, e2v_out, node_features=None, edge_weights=None, graph_feature=None):
    """
    Lets you define your oriented hypergraph with edges-to-vertices, which is one-to-one with the signed incidence
    :param num_nodes:
    :param e2v_in:
    :param e2v_out:
    :param node_features:
    :param edge_weights:
    :param graph_feature:
    :return:
    """
    return OrientedHypergraph(e2v_to_H(num_nodes, e2v_in, e2v_out), node_features=node_features,
                              edge_weights=edge_weights, graph_feature=graph_feature)


def e2v_to_v2e(num_node, e2v_in, e2v_out):
    v2e_in, v2e_out = [[] for i in range(num_node)], [[] for i in range(num_node)]
    for m in range(len(e2v_in)):
        for node in e2v_in[m]:
            v2e_in[node].append(m)
        for node in e2v_out[m]:
            v2e_out[node].append(m)
        # if m % (len(e2v_in) // 10):
        #     print(f"e2v_to_v2e edge {m}/{len(e2v_in)} done")
    for n in range(len(v2e_in)):
        v2e_in[n] = jnp.array(v2e_in[n])
        v2e_out[n] = jnp.array(v2e_out[n])
    return v2e_in, v2e_out


def calcDegreeDiag(num_node, v2e_in, v2e_out, W):
    D_in, D_out = np.zeros((2, num_node))
    for n in range(num_node):
        if len(v2e_in[n]) == 0:
            D_in[n] = 0
        else:
            D_in[n] = W[v2e_in[n]].sum()
        if len(v2e_out[n]) == 0:
            D_out[n] = 0
        else:
            D_out[n] = W[v2e_out[n]].sum()
    return jnp.array(D_in), jnp.array(D_out)


def calcDegreeDiag_from_e2v(num_node, e2v_in, e2v_out, W):
    D_in, D_out = np.zeros((2, num_node))
    for m in range(len(e2v_in)):
        D_in[e2v_in[m]] += W[m]
        D_out[e2v_out[m]] += W[m]
    return jnp.array(D_in), jnp.array(D_out)


def calcEdgeDegreeDiag(num_edge, e2v_in, e2v_out):
    E_in, E_out = np.zeros((2, num_edge))
    for m in range(num_edge):
        E_in[m] = len(e2v_in[m])
        E_out[m] = len(e2v_out[m])
    return jnp.array(E_in), jnp.array(E_out)


def calc_e2v2v_bool(num_edge, num_node, e2v_in, e2v_out):
    v2v_bool = np.zeros((num_edge, num_node, num_node))
    for m in range(num_edge):
        for i in e2v_in[m]:
            for j in e2v_out[m]:
                v2v_bool[m, i, j] = 1
    return jnp.array(v2v_bool)


def calc_v2v_bool(num_node, e2v_in, e2v_out):
    v2v = np.zeros((num_node, num_node))
    for m in range(len(e2v_in)):
        for i in e2v_in[m]:
            for j in e2v_out[m]:
                v2v[i, j] = 1
    return jnp.array(v2v)


def calc_v2v_bool_sparse(e2v_in, e2v_out):
    senders, receivers = [], []
    for m in range(len(e2v_in)):
        for n_in in range(len(e2v_in[m])):
            for n_out in range(len(e2v_out[m])):
                senders += [e2v_in[m][n_in]]
                receivers += [e2v_out[m][n_out]]
    # permute_sort = np.argsort(senders)
    # permute_sort_inv = np.empty_like(permute_sort)
    # permute_sort_inv[permute_sort] = np.arange(permute_sort.size)
    #
    # def unique_rows(a):
    #     a = np.ascontiguousarray(a)
    #     unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))
    #     return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
    #
    #
    inout = np.asarray([senders, receivers])
    # inout_unique = unique_rows(inout)
    # ordered_sender, ordered_receiver = senders[permute_sort], receivers[permute_sort]
    return inout


def flatten_list_of_arrays(x2y):
    x2y_flattened = [x for xs in x2y for x in xs]
    return jnp.asarray(x2y_flattened)


def calc_PLaplace_indexes_for(e2v_in, e2v_out):
    """
    Calculates all flattened indexes for the pairwise parallel transports and other vmap operations
    :param e2v_in: List of M arrays of edge input-sets
    :param e2v_out: List of M arrays of edge input-sets
    :return: List of indexes arrays
    """
    indexes_logs_edges_out, indexes_logs_nodes_in, indexes_logs_nodes_out = [], [], []
    indexes_PT_edges_in, indexes_PT_nodes_in, indexes_PT_nodes_out, indexes_PT_edges_segsum = [], [], [], []
    indexes_P_edges_in_div = []
    new_index = -1
    PT_index = 0
    test = []
    for m in range(len(e2v_in)):
        if m == 0:
            pass
        else:
            PT_index += len(e2v_in[m-1])
        for i in range(len(e2v_in[m])):
            new_index += 1
            for j_out in range(len(e2v_out[m])):
                indexes_logs_edges_out.append(new_index)
                indexes_logs_nodes_in.append(int(e2v_in[m][i]))
                indexes_logs_nodes_out.append(int(e2v_out[m][j_out]))
            for j_in in range(len(e2v_in[m])):
                indexes_PT_edges_segsum.append(new_index)
                indexes_PT_edges_in.append(PT_index + j_in)
                indexes_PT_nodes_in.append(int(e2v_in[m][j_in]))
                indexes_PT_nodes_out.append(int(e2v_in[m][i]))
            indexes_P_edges_in_div.append(m)
    return [jnp.asarray(indexes_logs_edges_out), jnp.asarray(indexes_logs_nodes_in), jnp.asarray(indexes_logs_nodes_out),
            jnp.asarray(indexes_PT_edges_in), jnp.asarray(indexes_PT_nodes_in), jnp.asarray(indexes_PT_nodes_out),
            jnp.asarray(indexes_P_edges_in_div), jnp.asarray(indexes_PT_edges_segsum)]


def e2v_to_H(n_node, e2v_in, e2v_out):
    n_edge = len(e2v_in)
    H = np.zeros((n_node, n_edge))
    for m in range(n_edge):
        for node_in in e2v_in[m]:
            H[node_in, m] = 1
        for node_out in e2v_out[m]:
            assert H[node_out, m] != 1, "e_in and e_out not unique"
            H[node_out, m] = -1
    return H


def H_to_e2v(H):
    e_list_in, e_list_out = [], []
    for m_edge in H.T:
        edge_in, edge_out = [], []
        for n, n_node in enumerate(m_edge):
            if n_node == 1:
                edge_in.append(n)
            elif n_node == -1:
                edge_out.append(n)
        e_list_in.append(edge_in)
        e_list_out.append(edge_out)
    return e_list_in, e_list_out


# def generate_random_edges(H: dhg.Hypergraph, num_edges: int = 1):
#     for m in range(num_edges):
#         size = np.random.randint(2, H.num_v)
#         h_edge = np.random.choice(np.arange(H.num_v), size=size, replace=False)
#         H.add_hyperedges([h_edge], e_weight=None, merge_op='mean', group_name='main')
#     return jnp.array(H)


def generate_random_signed_incidence(num_nodes, num_edges, sym=False, print_out=False, min_tot_edge_degree=3):
    vals = np.array([-1, 0, 1])

    for i in range(100):
        H = np.random.choice(vals, size=(num_nodes, num_edges), p=[0.3, 0.4, 0.3])
        if (np.prod(np.diag(H@H.T)) != 0 and (np.abs(H).sum(axis=0) >= min_tot_edge_degree).all() and
                ((H > 0).any(axis=0).all()) and ((H < 0).any(axis=0).all())) and ((H.T > 0).any(axis=0).all()):
            if print_out:
                print("H is valid incident")
            break
    if sym:
        H = np.concatenate((H, -H), axis=1)

    return jnp.array(H)


def generate_random_signed_incidence_Forward(num_nodes, num_edges, print_out=False, min_tot_edge_degree=2):
    vals = np.array([0, -1])
    for i in range(1000):
        H = np.random.choice(vals, size=(num_nodes, num_edges), p=[0.6, 0.4])
        input = np.random.randint(0, num_nodes, size=(num_edges))
        for j in range(num_edges):
            H[input[j], j] = 1
        if (np.prod(np.diag(H@H.T)) != 0 and (np.abs(H).sum(axis=0) >= min_tot_edge_degree).all() and
                ((H > 0).any(axis=0).all()) and ((H < 0).any(axis=0).all())) and ((H.T > 0).any(axis=0).all()):
            if print_out:
                print("H is valid incident")
            break
    assert i < 1000, "iter number exceeded, H is not valid"
    return jnp.array(H)


def generate_random_symmetric_signed_incidence(num_nodes, num_edges):
    vals = np.array([-1, 0, 1])
    H = np.random.choice(vals, size=(num_nodes, num_edges), p=[0.2, 0.6, 0.2])
    for i in range(100):
        if (np.prod(np.diag(H@H.T)) != 0 and (np.abs(H).sum(axis=0) >= 2).all() and
                ((H > 0).any(axis=0).all()) and ((H < 0).any(axis=0).all())):
            print("H is valid incident")
            break
        H = np.random.choice(vals, size=(num_nodes, num_edges), p=[0.2, 0.6, 0.2])
    return jnp.array(H)


def orient_hypergraph_randomly(e2v):
    e2v_in, e2v_out = [], []
    for m_edge, edge in enumerate(e2v):
        cuts = np.random.randint(1, len(edge) - 1)
        np.random.shuffle(edge)
        e2v_in.append(np.sort(edge)[0:cuts])
        e2v_out.append(np.sort(edge)[cuts:])
    return e2v_in, e2v_out


def orient_hypergraph_one_to_all(e2v):
    e2v_in, e2v_out = [], []
    for m_edge, edge in enumerate(e2v):
        for n, v in enumerate(edge):
            e2v_in.append([v])
            if n == len(edge) - 1:
                e2v_out.append(edge[0:n])
            else:
                e2v_out.append(edge[0:n] + edge[n+1:])
    return e2v_in, e2v_out


# def OH_to_dhg(OH: OrientedHypergraph):
#     e2v = []
#     for edge in OrientedHypergraph(np.abs(OH.H)).e2v_in:
#         neihboorhood_list = []
#         for i in edge:
#             neihboorhood_list.append(int(i))
#         e2v.append(neihboorhood_list)
#     return dhg.Hypergraph(len(OH.X), e2v)


def OHtuple_to_jraph_edgeless(OH: OHGraphTupleReduced, keep_edges=False):
    """
    Converts OrientedHypergraphTuple to jraph tuple with empty edges.
    :param keep_edges: Initialize edges as well. NB: Only for hypergraphs with edges size 2(graphs)
    :param OH:
    :return:
    """
    if keep_edges:
        graph = jraph.GraphsTuple(
          nodes=OH.X,
          edges=OH.W,
          senders=jnp.array(OH.e2v_in).reshape(len(OH.e2v_in)),
          receivers=jnp.array(OH.e2v_out).reshape(len(OH.e2v_out)),
          n_node=jnp.asarray(OH.num_node)[None],
          n_edge=jnp.asarray(OH.num_edge)[None],
          globals=OH.globals
          )
    else:
        graph = jraph.GraphsTuple(
          nodes=OH.X,
          edges=OH.W,
          senders=jnp.array([]),
          receivers=jnp.array([]),
          n_node=jnp.asarray(OH.num_node)[None],
          n_edge=jnp.asarray(OH.num_edge)[None],
          globals=OH.globals
          )
    return graph


# def dhg_to_OH(UH: dhg.Hypergraph, random=True):
#     """WARNING: This function is by default stochastic due to OH having more information as compared to UH"""
#     UH_e2v_tensor = H_to_e2v(np.array(UH.H_e2v.to_dense()))[0]
#     if random:
#         e2v_in, e2v_out = orient_hypergraph_randomly(UH_e2v_tensor)
#     else:
#         e2v_in, e2v_out = orient_hypergraph_one_to_all(UH_e2v_tensor)
#     return OrientedHypergraph_e2v_struct(UH.num_v, e2v_in, e2v_out)


# def draw_OH_as_dhg(OH: OrientedHypergraph):
#     """Uses the unoriented drawer to draw the OH without including vertex orientations in edges"""
#     dhg.visualization.draw_hypergraph(OH_to_dhg(OH), v_label=np.arange(len(OH.X)))
#     plt.show()


def jraph_to_OH(graph: jraph.GraphsTuple):
    return OrientedHypergraph_e2v_struct(len(graph.nodes), graph.senders.reshape((len(graph.senders), 1)),
                                         graph.receivers.reshape((len(graph.receivers), 1)), node_features=graph.nodes,
                                         edge_weights=graph.edges, graph_feature=graph.globals)


def clique_expand(OH: OrientedHypergraph, intra_connect=False):
    """
    Clique expand oriented hypergraph.
    :param OH: Oriented hypergraph class or named tuple
    :param intra_connect: Makes cliques ouf of inputs and outputs. No functionality yet.
    :return:
    """
    senders, receivers, weights = [], [], []
    skip = False
    for m in range(OH.num_edge):
        print(f"Edge {m}/{OH.num_edge} click expanded")
        for n_in in range(len(OH.e2v_in[m])):
            for n_out in range(len(OH.e2v_out[m])):
                for send_num, i in enumerate(senders):
                    if OH.e2v_in[m][n_in] == i:
                        if OH.e2v_out[m][n_out] == receivers[send_num]:
                            weights[send_num] += OH.W[m]
                            skip = True

                if skip:
                    skip = False
                else:
                    senders += [OH.e2v_in[m][n_in]]
                    receivers += [OH.e2v_out[m][n_out]]
                    weights += [OH.W[m]]

    senders, receivers, weights = np.array(senders), np.array(receivers), np.array(weights)
    print(f"Biggest edge weight is {weights.max()}")
    return OrientedHypergraph_e2v_struct(OH.num_node, senders.reshape((len(senders), 1)),
                                         receivers.reshape((len(receivers), 1)), node_features=OH.X,
                                         edge_weights=weights, graph_feature=OH.globals)


if __name__ == '__main__':
    N = 6
    M = 3

    # OH1 = OrientedHypergraph(generate_random_signed_incidence(N, M))
    OH1 = OrientedHypergraph(generate_random_signed_incidence(N, M), edge_weights=1/N)
    # dhg1 = OH_to_dhg(OH1)
    # OH_1_to_all = dhg_to_OH(dhg1, random=False)
    # draw_OH_as_dhg(OH_1_to_all)
    OH_red_tuple = OrientedHypergraphTuple_e2v_struct(OH1.num_node, OH1.e2v_in, OH1.e2v_out)
    pass




