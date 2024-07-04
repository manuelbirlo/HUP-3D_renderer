# Use a CUDA-enabled Ubuntu base image
FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04

WORKDIR /root

ENV PATH="/usr/local/cuda-11.8/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"

# Set the environment variable for the display
ENV DISPLAY=host.docker.internal:0.0

# Install dependencies for Conda and pip
RUN apt-get update && \
    apt-get install -y wget git libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Install Blender-related dependency
RUN apt-get update && apt-get install -y libglib2.0-0

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm -f Miniconda3-latest-Linux-x86_64.sh

# Set path for Conda
ENV PATH /root/miniconda3/bin:$PATH