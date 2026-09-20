# From conda environment for CUDA 12.1 with PyTorch and DGL
FROM ubuntu:22.04

# Define the type of build (cuda121 or cpu)
ARG TYPE=cuda121

# Set the working directory
WORKDIR /app

# Install conda
RUN apt-get update && apt-get install -y wget bzip2 git build-essential make psmisc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

    # Copy the environment file
COPY docker/env-${TYPE}.yml env.yml

RUN /opt/conda/bin/conda env create -f env.yml && \
    /opt/conda/bin/conda init && \
    /opt/conda/bin/conda clean -a -y

ENV PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

COPY . .

RUN conda run -n llm-dgl-${TYPE} python3 launch.py --help

    # Activate the environment
SHELL ["conda", "run", "-n", "llm-dgl-${TYPE}", "/bin/bash", "-c"]
