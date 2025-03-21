from typing import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.scipy.sparse.linalg import bicgstab
from morphomatics.manifold import Manifold, PowerManifold

# from morphomatics.graph.operators import mfdg_laplace
from morphomatics.nn.tangent_layers import TangentMLP
from morphomatics.opt import RiemannianNewtonRaphson

from OrientedHypergraphs.objects import OHGraphTupleReduced, OrientedHypergraph
from OrientedHypergraphs.operators import FLaplace, PLaplace

"""This file is temporarly, as the imported flax is implemended with haiku syntax"""


class flow_layer(nn.Module):
    """
    Graph flow layer for graphs with manifold-valued features. The flow equation is integrated explicitly by default,
    but an implicit scheme is also available.

    See

    M. Hanik, G, Steidl, C. v. Tycowicz. "Manifold GCN: Diffusion-based Convolutional Neural Network for
    Manifold-valued Graphs" (https://arxiv.org/abs/2401.14381)

    for a detailed description.

    """

    def __init__(self, M: Manifold, n_steps=1, implicit=False, name=None):
        """
        :param M: manifold in which the features lie
        :param n_steps: number of explicit steps to approximate the flow with explicit Euler
        :param implicit: whether to use implicit or explicit Euler integration
        :param name: layer name (see haiku documentation of haiku)
        """
        super().__init__(name=type(self).__name__ if name is None else name)
        self.M = M
        self.n_steps = n_steps
        self.step = self._single_euler_step
        # self.step = self._implicit_euler_step if implicit else self._single_euler_step

    def _single_euler_step(
        self, OH: OHGraphTupleReduced, time: jnp.ndarray, delta: jnp.ndarray
    ) -> OHGraphTupleReduced:
        """Single step of the explicit Euler method for diffusion

        :param G: graph with manifold valued vectors as features; length of vector must equal the flow layer width
        :param time: vector of time parameters (same length as number of features)
        :param delta: vector of "minimal step sizes"
        :return: updated graph
        """

        def _multi_laplace(channel):
            return PLaplace(OH._replace(X=channel), self.M)

        def _activation(feature, vector, d):
            nrm = jnp.sqrt(
                self.M.metric.inner(feature, vector, vector)
                + jnp.finfo(jnp.float64).eps
            )
            act = jax.nn.sigmoid(nrm - d)
            return jax.lax.cond(
                nrm * act >= 1e-3,
                lambda a, w: a * w,
                lambda _, w: jnp.zeros_like(w),
                act,
                vector,
            )

        v = jax.vmap(_multi_laplace, in_axes=1, out_axes=1)(OH.X)

        # ReLU-type activation
        delta = jnp.stack(
            [
                delta,
            ]
            * v.shape[0]
        )
        v = jax.vmap(jax.vmap(_activation))(OH.X, v, delta)

        v = -v * time.reshape((1, -1) + (1,) * (v.ndim - 2))
        x = jax.vmap(jax.vmap(self.M.connec.exp))(OH.X, v)
        return OH._replace(X=x)

    # def _implicit_euler_step(self, OH: OHGraphTupleReduced, time: jnp.ndarray, delta=None) -> OHGraphTupleReduced:
    #     """Single step of the implicit Euler method for diffusion
    #
    #     :param G: graph with manifold valued vectors as features; length of vector must equal the flow layer width
    #     :param time: vector of time parameters (same length as number of features)
    #     :param delta: only needed for out of syntax reasons
    #     :return: updated graph
    #     """
    #     # n_nodes x n_channels x point_shape
    #     n, c, *shape = OH.X.shape
    #
    #     # power manifold
    #     P = PowerManifold(self.M, n*c)
    #
    #     # current state
    #     x_cur = OH.X.reshape(-1, *shape)
    #
    #     # root of F characterizes solution to implicit Euler step
    #     def F(x: jnp.array):
    #         L = lambda a: PLaplace(OH._replace(X=a), self.M)
    #         Lx = jax.vmap(L, in_axes=1, out_axes=1)(x.reshape(n, c, *shape))
    #         tLx = Lx * time.reshape((1, -1) + (1,)*len(shape))
    #         diff = P.connec.log(x, x_cur)
    #         return diff - tLx.reshape(-1, *shape)
    #
    #     # x_next = RiemannianNewtonRaphson.solve(P, F, x_cur, maxiter=1)
    #     ###############################
    #     # unroll single interation
    #     ###############################
    #     # solve for update direction: v = -J⁻¹F(x)
    #     J = lambda v: jax.jvp(F, (x_cur,), (v,))[1]
    #     v, _ = bicgstab(J, -F(x_cur))
    #     # step
    #     x_next = P.connec.exp(x_cur, v)
    #
    #     return OH._replace(X=x_next.reshape(n, c, *shape))

    def __call__(self, OH: OHGraphTupleReduced) -> OHGraphTupleReduced:
        """
        :param G: graphs tuple with features of shape: num_nodes * num_channels * point_shape
        :return: graphs tuple with features of shape: num_nodes * num_channels * point_shape

        Apply discretized diffusion (with final activation) flow to each channel
        """

        width = OH.X.shape[1]  # number of channels
        # init parameter
        t_init = nn.initializers.TruncatedNormal(stddev=1, mean=1)
        delta_init = nn.initializers.TruncatedNormal(stddev=1, mean=1)
        ####################
        t = nn.get_parameter("t_sqrt", shape=[width], init=t_init)
        delta = nn.get_parameter("delta_sqrt", shape=[width], init=delta_init)
        ####################

        # print(t)

        # map to non-negative weights
        t = t**2
        delta = delta**2

        # make n_steps explicit Euler steps for each graph in the batch
        # for _ in range(self.n_steps):
        #     G = self.step(G, t / self.n_steps)

        def step(graph, _):
            graph = self.step(graph, t / self.n_steps, delta)
            return graph, None

        OH, _ = jax.lax.scan(step, OH, None, self.n_steps, unroll=self.n_steps)

        return OH
