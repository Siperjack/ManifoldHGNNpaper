import random
# from typing import Any, List, NamedTuple, Tuple
from typing import Any
import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph
import matplotlib.pyplot as plt
import numpy as np
import optax
from morphomatics.graph.operators import max_pooling, mean_pooling
from morphomatics.manifold import Manifold, Sphere
from morphomatics.nn import MfdFC, MfdInvariant
from morphomatics.nn import flow_layer as FlowLayer
from mpl_toolkits.axes_grid1 import make_axes_locatable

from OrientedHypergraphs.objects import (
    OHGraphTupleReduced,
    OHtuple_to_jraph_edgeless,
    # OrientedHypergraph,
    jraph_to_OH,
)
from OrientedHypergraphs.train import TrainingState, evaluate, evaluate_F1, update
from OrientedHypergraphs.utils import pickle_dump, pickle_load
from toy_examples.flow_sphere.example_HGs import (
    # OH_rectangle,
    irregular_graph,
    random_S2_OH,
    # rectangle,
    tetrahedron,
)

NUM_CLASSES = 3


def parameter_dashboard(
    t_hist,
    invariant_MfdFC_hist,
    invariant_linear_hist,
    train_accuracies,
    test_accuracies,
    train_accuracies_acc,
    test_accuracies_acc,
    l,
    save_fig_as=False,
):
    """
    Plots training/testing diagrams
    """
    x = np.arange(len(t_hist))

    fig, axs = plt.subplots(3, 2, figsize=(8, 8))
    fig.suptitle(
        f"{NUM_CLASSES}-Classes Generating-Algorithm Classification", fontsize="large"
    )
    gs = axs[2, 0].get_gridspec()
    for ax in axs[-1, :]:
        ax.remove()
    axbig = fig.add_subplot(gs[-1, :])
    N_flowparameters = min(len(t_hist[0]), 10)
    for i in range(N_flowparameters):  # Cap at 10 in plot
        axs[0, 0].plot(x, np.array(t_hist)[:, i], label=f"$t_{i}$")
    axs[0, 0].set_title("Flow Layer")
    axs[0, 0].set_xlabel(f"optimizations steps/{l}")
    axs[0, 0].set_ylabel("parameter value")
    # axs[0].grid(True)
    axs[0, 0].legend()

    invariant_MfdFC_hist = np.array(invariant_MfdFC_hist)

    ma = axs[1, 0].matshow(invariant_MfdFC_hist[-1], interpolation="none")
    axs[1, 0].set_title("MfdFC Layer in Invariant Block")
    divider = make_axes_locatable(axs[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(ma, cax=cax)

    invariant_linear_hist = np.array(invariant_linear_hist)

    mb = axs[0, 1].matshow(invariant_linear_hist[-1], interpolation="none")
    divider = make_axes_locatable(axs[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(mb, cax=cax)
    axs[0, 1].set_title("Linear Layer in Invariant Block")
    axs[1, 1].set_title("Final Linear Layer")
    train_accuracies_acc = np.array(train_accuracies_acc)
    test_accuracies_acc = np.array(test_accuracies_acc)
    axs[1, 1].plot(train_accuracies_acc, label="training")
    axs[1, 1].plot(test_accuracies_acc, label="test")
    axs[1, 1].set_title("Prediction Results")
    axs[1, 1].set_xlabel(f"optimizations steps/{l}")
    axs[1, 1].set_ylabel("acc score")
    axs[1, 1].set_ylim([0, 1])
    axs[1, 1].legend()

    train_accuracies = np.array(train_accuracies)
    test_accuracies = np.array(test_accuracies)
    axbig.plot(train_accuracies, label="training")
    axbig.plot(test_accuracies, label="test")
    axbig.set_title("Prediction Results")
    axbig.set_xlabel(f"optimizations steps/{l}")
    axbig.set_ylabel("F1 macro score")
    axbig.set_ylim([0, 1])
    axbig.legend()

    fig.tight_layout()
    plt.show()

    if save_fig_as:
        fig.savefig(save_fig_as)


class GraphClassifier(nn.Module):
    width: int
    num_classes: int
    dropout_rate: float = 0.0
    n_steps: int = 1
    simple_network: bool = True
    invariant_nc: int = 2
    M: Any  # The manifold
    step_type: str = "explicit_flow"

    @nn.compact
    def __call__(self, OH: OHGraphTupleReduced, train: bool = True) -> jnp.ndarray:
        z = OH.X
        # Makes WIDTH amount of channels
        OH = OH._replace(
            X=jnp.concatenate(
                [
                    z[:, None, :],
                ]
                * self.width,
                axis=1,
            )
        )

        # Diffusion layer
        OH = FlowLayer(self.M, implicit=self.step_type, n_steps=self.n_steps)(OH)

        # MfdFC block
        if not self.simple_network:
            z = MfdFC(self.M, self.width)(OH.X[None])[0]
            OH = OH._replace(X=z)
            OH = FlowLayer(self.M, implicit=self.step_type, n_steps=self.n_steps)(OH)

        # Invariant
        z = OH.X
        z = MfdInvariant(self.M, self.width, nC=self.invariant_nc)(z[None])[0]

        # Nonlinear
        z = nn.leaky_relu(z)
        if self.dropout_rate > 0 and train:
            z = nn.Dropout(rate=self.dropout_rate)(z, deterministic=not train)

        # Max pooling per graph
        G = OHtuple_to_jraph_edgeless(OH)
        G = jraph.batch([G])
        z = jnp.concatenate((max_pooling(G, z), mean_pooling(G, z)), axis=1)

        # MLP for classification
        z = nn.Dense(features=self.width)(z)
        z = nn.leaky_relu(z)
        z = nn.Dense(features=self.width // 2)(z)
        z = nn.leaky_relu(z)
        z = nn.Dense(features=self.num_classes)(z)

        return z


def main(
    num_node: int,
    graph_num: int,
    n_train: int,
    n_epochs: int,
    M: Manifold,
    simple_network: bool = True,
    step_type: str = "explicit_flow",
    batch_padding: bool = False,
    batch_size: int = 1,
    learningrate: float = 1e-3,
    load_device: str = "gpu",
    save_fig_as: str = False,
    save_data_file_name: str = False,
    load_data_file_name: str = False,
):
    """
    num_node: #nodes of input graphs
    graph_num: The total number of triples of graphs generated. True number of graphs is three times this.
    batch_size: #graphs in each batch
    n_train: #update steps per epoch.
    n_epochs: #epochs
    space: string of embedded manifold. Supported is "Euclidean", "hyperbolic" or "SPD"
    load_graphs: A file name of a file containing graphs embedded in a manifold supported. In current version
                generating graph directly is not supported. Use "generate_graph_morphomatics" to generate files
    step_type: "implicit_flow" will make the flow layer use implicit Euler iteration
    batch_padding: If yes all dummy nodes dummy edges and dummy graphs are added such that all resulting batches
                    are initialized with the same shape overall. Should improve runtime, testing shows it often doesn't.
    pre_batched: If data is already batched in jraph format then "True" makes in not attempt to batch again
    M_structure: Lets user choose another metric from the default. Sure to be supported: Affine-Invariant for SPD
    degree_embed: The dimension of the degree embedding. If "False" the data is assumed to be embedded in M already
    learningrate: Change learningrate from function call
    load_device: Whether to load the data into the cpu or the gpu
    save_fig_as: A file name to save plot of training/testing to. Default is not to save.
    """

    if load_data_file_name:
        data = pickle_load(load_data_file_name)
        print("data loaded")
    else:
        data = []
        for i in range(graph_num):
            OH_1 = jraph_to_OH(
                tetrahedron(homogeneous_edge_weights=True)
            ).OHGraphTupleReduced._replace(globals=jnp.array([0]))
            OH_2 = jraph_to_OH(
                irregular_graph(homogeneous_edge_weights=True)
            ).OHGraphTupleReduced._replace(globals=jnp.array([1]))
            OH_3 = jraph_to_OH(
                irregular_graph(homogeneous_edge_weights=False)
            ).OHGraphTupleReduced._replace(globals=jnp.array([2]))
            # OH_1 = tetrahedron(homogeneous_edge_weights=True)._replace(globals=jnp.array([0], int))
            # OH_2 = irregular_graph(homogeneous_edge_weights=True)._replace(globals=jnp.array([1], int))
            # OH_3 = rectangle(homogeneous_edge_weights=True)._replace(globals=jnp.array([2], int))
            # OH_3 = jraph_to_OH(rectangle(homogeneous_edge_weights=True)).OHGraphTupleReduced._replace(
            #     globals=jnp.array([2]))
            # OH_3 = random_S2_OH(num_node, num_node, scale=0.1).OHGraphTupleReduced._replace(
            #     globals=jnp.array([2]))

            OH_1 = random_S2_OH(
                num_node, num_node, scale=0.01
            ).OHGraphTupleReduced._replace(globals=jnp.array([0]))
            OH_2 = random_S2_OH(
                num_node, num_node, scale=0.1
            ).OHGraphTupleReduced._replace(globals=jnp.array([1]))
            OH_3 = random_S2_OH(
                num_node, num_node, scale=1.0
            ).OHGraphTupleReduced._replace(globals=jnp.array([2]))
            data.append(OH_1)
            data.append(OH_2)
            data.append(OH_3)
            # data.append(jraph.batch([OH_1]))
            # data.append(jraph.batch([OH_2]))
            # data.append(jraph.batch([OH_3]))

            # data.append(jraph.batch([OH_1, OH_2, OH_3]))
            # batch_size = 3

        print("toy data generated")
        if save_data_file_name:
            pickle_dump(data, save_data_file_name)
            print("graph image saved")

    """Network parameters"""
    WIDTH = 8
    dropout = 0
    n_steps = 1
    simple_network = True
    invariant_nC = 2

    print(
        f"Parameters of network are: \n WIDTH = {WIDTH}"
        f"\n learningRate = {learningrate}\n n_steps = {n_steps}"
        f"\n simple_network = {simple_network}, dropout = {dropout}"
        f"\n nC of invariant = {invariant_nC}"
        f"\n NUM CLASSES = {NUM_CLASSES}"
    )

    # Initialize the model
    model = GraphClassifier(
        width=WIDTH,
        num_classes=NUM_CLASSES,
        dropout_rate=dropout,
        n_steps=n_steps,
        simple_network=simple_network,
        invariant_nc=invariant_nC,
        M=M,
    )

    # Initialize parameters
    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)
    params = model.init(init_key, data[0])

    # Setup optimizer
    optimizer = optax.adam(learningrate)
    opt_state = optimizer.init(params)
    state = TrainingState(params, params, opt_state)

    # Create train/test split
    random.shuffle(data)
    split_idx = int(0.8 * len(data))  # 80-20 split
    train_dataset = data[:split_idx]
    test_dataset = data[split_idx:]

    t_hist = [params["flow_layer"]["t_sqrt"] ** 2]
    invariant_MfdFC_hist = [params["MfdInvariant/~/MfdFC"]["w"]]
    invariant_linear_hist = [params["MfdInvariant/~/linear"]["w"]]
    train_accuracies_F1 = []
    test_accuracies_F1 = []
    train_accuracies_acc = []
    test_accuracies_acc = []
    evaluate_counter = 0
    for epoch in range(n_epochs):
        # Training & evaluation loop.
        for step in range(n_train):
            if batch_padding:
                mask = jraph.get_graph_padding_mask(train_dataset[step])
            else:
                mask = jnp.ones(batch_size) > 0  # [True, ..., True]
            if epoch == 0 and step == 0:
                print(f"mask is: {mask}")
                print("First state updating...")
            state = update(
                state,
                train_dataset[step],
                train_dataset[step].globals,
                optimizer,
                model,
                key,
                mask,
            )
            if epoch == 0 and step == 0:
                print("First state updated")
            evaluate_interval = int(
                n_train / 2
            )  # Calculate scores over full training and testing sets twice per epoch
            if (step + evaluate_interval) % (evaluate_interval + 1) == evaluate_interval:
                train_accuracy_F1 = jnp.array([0])
                test_accuracy_F1 = jnp.array([0])
                train_accuracy_acc = jnp.array([0])
                test_accuracy_acc = jnp.array([0])
                for g in train_dataset:
                    batch_accuracy_F1 = np.array(
                        evaluate_F1(
                            state.avg_params,
                            g,
                            g.globals,
                            NUM_CLASSES,
                            model,
                            key,
                            mask=mask,
                        )
                    )
                    batch_accuracy_acc = np.array(
                        evaluate(
                            state.avg_params,
                            g,
                            g.globals,
                            NUM_CLASSES,
                            model,
                            key,
                            mask=mask,
                        )
                    )
                    train_accuracy_F1 += batch_accuracy_F1
                    train_accuracy_acc += batch_accuracy_acc

                train_accuracy_F1 = train_accuracy_F1 / len(train_dataset)
                train_accuracies_F1.append(train_accuracy_F1)
                train_accuracy_acc = train_accuracy_acc / len(train_dataset)
                train_accuracies_acc.append(train_accuracy_acc)

                for g in test_dataset:
                    batch_accuracy_F1 = np.array(
                        evaluate_F1(
                            state.avg_params,
                            g,
                            g.globals,
                            NUM_CLASSES,
                            model,
                            key,
                            mask=mask,
                        )
                    )
                    batch_accuracy_acc = np.array(
                        evaluate(
                            state.avg_params,
                            g,
                            g.globals,
                            NUM_CLASSES,
                            model,
                            key,
                            mask=mask,
                        )
                    )

                    test_accuracy_F1 += batch_accuracy_F1
                    test_accuracy_acc += batch_accuracy_acc

                test_accuracy_F1 = test_accuracy_F1 / len(test_dataset)
                test_accuracies_F1.append(test_accuracy_F1)
                test_accuracy_acc = test_accuracy_acc / len(test_dataset)
                test_accuracies_acc.append(test_accuracy_acc)

                print(
                    {
                        "epoch.step": f"{epoch}.{step}",
                        "train_average_F1": f"{train_accuracy_F1[0]:.3f}",
                        "test_average_F1": f"{test_accuracy_F1[0]:.3f}",
                    }
                )
                print(
                    {
                        "epoch.step": f"{epoch}.{step}",
                        "train_average_acc": f"{train_accuracy_acc[0]:.3f}",
                        "test_average_acc": f"{test_accuracy_acc[0]:.3f}",
                    }
                )

                t_hist.append(np.array(state.params["flow_layer"]["t_sqrt"] ** 2))
                invariant_MfdFC_hist.append(
                    np.array(state.params["MfdInvariant/~/MfdFC"]["w"])
                )
                invariant_linear_hist.append(
                    np.array(state.params["MfdInvariant/~/linear"]["w"])
                )

                if evaluate_counter % int(n_epochs / 2) == 1:
                    parameter_dashboard(
                        t_hist,
                        invariant_MfdFC_hist,
                        invariant_linear_hist,
                        train_accuracies=train_accuracies_F1,
                        test_accuracies=test_accuracies_F1,
                        train_accuracies_acc=train_accuracies_acc,
                        test_accuracies_acc=test_accuracies_acc,
                        evaluate_interval=evaluate_interval,
                    )
                evaluate_counter += 1
    parameter_dashboard(
        t_hist,
        invariant_MfdFC_hist,
        invariant_linear_hist,
        train_accuracies=train_accuracies_F1,
        test_accuracies=test_accuracies_F1,
        train_accuracies_acc=train_accuracies_acc,
        test_accuracies_acc=test_accuracies_acc,
        evaluate_interval=evaluate_interval,
        save_fig_as=save_fig_as,
    )
    print(
        f"Last 10 steps of F1: (training, testing) = ({jnp.array(train_accuracies_F1)[-10:].sum() / 10:.3f}, "
        f"{jnp.array(test_accuracies_F1)[-10:].sum() / 10:.3f})."
    )
    print(
        f"Last 10 steps of acc: (training, testing) = ({jnp.array(train_accuracies_acc)[-10:].sum() / 10:.3f}, "
        f"{jnp.array(test_accuracies_acc)[-10:].sum() / 10:.3f})."
    )


if __name__ == "__main__":
    """Playground"""
    Man = Sphere()
    num_node = 4
    graph_num = 20  # 300
    n_train = int(4 / 5 * NUM_CLASSES * graph_num)
    assert (4 * NUM_CLASSES * graph_num / 5) == n_train
    n_epochs = 200

    # main(num_node, graph_num, n_train, n_epochs, M=Man, load_data_file_name=f"S2_toy_class_{num_node}_{graph_num}")
    main(num_node, graph_num, n_train, n_epochs, M=Man, learningrate=1e-5)
