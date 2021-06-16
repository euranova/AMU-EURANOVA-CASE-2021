# Use an anaconda runtime as a parent image
FROM nvidia/cuda:10.2-base-ubuntu18.04

RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
      g++ \
      git \
      wget \
      gcc \
      libgtk2.0-dev \
      bzip2 \
      curl \
      ca-certificates \
      graphviz \
      libsm6 \
      libxext6 \
      libxrender-dev \
      libgl1-mesa-glx \
      libhdf5-dev \
      openmpi-bin \
      && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH
ENV PYTHON_VERSION=3.7
ARG PYTHON_VERSION=3.7

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $HOME/miniconda.sh && \
    chmod 0755 $HOME/miniconda.sh && \
    $HOME/miniconda.sh -b -p $CONDA_DIR &&\
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh &&\
    rm $HOME/miniconda.sh

RUN conda install -y python=${python_version} \
    && conda config --append channels conda-forge

RUN mkdir event_extraction

WORKDIR /event_extraction

COPY . /event_extraction/

COPY . .

# Install any needed packages specified in pythonenv.yml
RUN conda env create -f environment.yml && conda clean -ay

# Pull the environment name out of the environment.yml
RUN echo "source activate $(head -1 environment.yml | cut -d' ' -f2)" > ~/.bashrc
ENV PATH /opt/conda/envs/bin:$PATH

# fix issue #39 of encoding error during run of docker container
ENV LC_ALL=C.UTF-8 LANG=C.UTF-8
