# Manifold-valued Graph Neural Networks

Repository for development of Jo master's thesis paper

## Dependencies

### Main Dependencies
- **JAX** (`jax`, `jaxlib`): For numerical computing and automatic differentiation
- **NumPy** (`numpy`): For numerical operations
- **Deep Hypergraph** (`dhg`): For hypergraph operations
- **Flax** (`flax`): Neural network library
- **imageio**: For visualization and GIF creation
- **Pillow**: For image processing
- **SciPy** (`scipy`): For scientific computing

### Development Dependencies
- **setuptools**: For package management
- **pip**: For package installation
- **wheel**: For building wheels

### Optional Dependencies
- **matplotlib**: For plotting
- **jupyter**: For running notebooks
- **pytest**: For testing


## Installation

### 1. Clone the repository:

bash
git clone https://github.com/yourusername/ManifoldHGNNpaper.git

### 2. Create and activate a virtual environment:

python -m venv venv

venv\Scripts\activate # On Windows

OR

source venv/bin/activate # On Unix/MacOS

### 3. Install requirements:

bash
pip install -r requirements.txt


### 4. Install in development mode:

bash
pip install -e .

The -e or develop flag means:

- Package is installed in "editable" mode
- Changes to source code take effect immediately
- No need to reinstall after code changes

Make sure your virtual environment is activated before running these commands!

### Alternative: Run in devcontainer with everythinng set up and run when ready. 
NB: This option often doesn't render gifs, and therefore lines creating gifs are commented out.

## HGNN paper:

Run toy_examples/flow_sphere/S2flow.py with different different parameters of the random_S2_OH(args) function.
