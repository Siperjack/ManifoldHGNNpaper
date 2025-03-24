FROM mcr.microsoft.com/devcontainers/python:3.11

# Keep the container lean
ENV DEBIAN_FRONTEND=noninteractive

# Basic system setup
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*


# The rest (Python packages, system libraries) will be handled by:
# - features (python and common-utils)
# - postCreateCommand (OpenGL libraries and pip install)
