import pickle
import imageio

from OrientedHypergraphs.utils import animate_S2_valued_OH
file_name = f'./gifs/laplacian_timeseries'
#file_name = f'laplacian_timeseries'
with open(file_name, 'rb') as file:
    X_F_t, X_P_t, X_click_t = pickle.load(file)
print(f"graph image {file_name} loaded")

"""
Animate the Fréchet mean, pairwise and reduced Laplace timeseries with the lines that follows. Only one runs at a time.
"""
animate_S2_valued_OH(X_F_t[:len(X_F_t)//2:5], gif_name="./gifs/test_F.gif")
animate_S2_valued_OH(X_P_t[::10], gif_name="./gifs/test_P.gif")
animate_S2_valued_OH(X_click_t[0:100], gif_name="./gifs/test_C.gif")