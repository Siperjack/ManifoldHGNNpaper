import pyvista as pv
from OrientedHypergraphs.objects import OrientedHypergraph
import pickle
#  import imageio

def draw_S2_valued_OH(OH: OrientedHypergraph):
    """Draw set of S2-valued splines NB: not yet with hyperedges
    :param OH: Oriented hypergraph with S2-valued features.
    """
    PP = pv.Plotter(shape=(1, 1), window_size=[1800, 1500])

    sphere = pv.Sphere(1)
    PP.add_mesh(sphere, color="#F5FFFF", show_edges=True)

    C = OH.X.reshape((-1, 3))
    for c in C:
        PP.add_mesh(pv.Sphere(radius=0.02, center=c), color="r")

    PP.show_axes()
    PP.show()
    PP.close()


def animate_S2_valued_OH(X_t, gif_name="test.gif"):
    """Draw set of S2-valued splines NB: not yet with hyperedges
    :param X_t: timeseries of S2-valued node configurations
    :param gif_name: the name of the file to save the gif
    """
    # PP = pv.Plotter(shape=(1, 1), window_size=[1800, 1500])
    frame_rate = len(X_t) // 3
    PP = pv.Plotter(notebook=False, off_screen=False)
    PP.open_gif(gif_name, fps=frame_rate)


    sphere = pv.Sphere(1)
    PP.add_mesh(sphere, color="#F5FFFF", show_edges=True)

    C = X_t[0].reshape((-1, 3))
    actors = []
    for c in C:
        mesh = pv.Sphere(radius=0.02, center=c)
        PP.add_mesh(mesh, color="r")
        actors.append(mesh)

    for num in range(1, len(X_t)):  # start from 1 as we already plotted the first frame
        # Update each body part
        # mean_position = np.array([0.0, 0.0, 0.0])
        for node_num, c in enumerate(X_t[num]):
            # x = df.loc[num, f'{body_part}_x']
            # y = df.loc[num, f'{body_part}_y']
            # z = df.loc[num, f'{body_part}_z']
            # mesh.points = pv.Sphere(radius=5, center=(x, y, z)).points  # Update points of existing mesh
            # mean_position += np.array([x, y, z])
            actors[node_num].points = pv.Sphere(radius=0.02, center=c).points
        PP.write_frame()
    # def callback(step):
    #     for actor in actors:
    #         actor.position = [step / 100.0, step / 100.0, 0]
    #
    # PP.add_timer_event(max_steps=200, duration=500, callback=callback)
    # PP.show_axes()
    # # PP.show()
    # # PP.close()
    # cpos = [(0.0, 0.0, 10.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    # PP.show(cpos=cpos)
    PP.close()
    return None


def pickle_load(file_name):
    with open(file_name, 'rb') as in_file:
        return pickle.load(in_file)


def pickle_dump(data, file_name):
    with open(file_name, 'wb') as in_file:
        return pickle.dump(data, in_file)