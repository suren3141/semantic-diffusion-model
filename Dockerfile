# Build : docker build -t semantic-diffusion-model -f environment/Dockerfile .

# Run :   docker run --gpus all --ipc host --rm -it \
#         -v <PATH_TO_DATASET>:/mnt/dataset \
#         --name semantic-diffusion-model semantic-diffusion-model bash

ARG PYTORCH="1.12.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

## To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        libjpeg-dev \
        libpng-dev && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 wget nano unzip libopenmpi-dev \
    && apt-get clean

# Create the environment:
#COPY environment .
#RUN conda env create -f environment/environment.yaml

#RUN conda create -n cross_image python==3.10 -y 

# Make RUN commands use the new environment:
#SHELL ["conda", "run", "-n", "cross_image", "/bin/bash", "-c"]

## Already installed, so delete this
#RUN pip install --no-input pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7

WORKDIR /workspace
RUN git clone https://github.com/openai/guided-diffusion.git
WORKDIR /workspace/guided-diffusion
RUN pip install -e .

WORKDIR /workspace
RUN git clone https://github.com/suren3141/semantic-diffusion-model.git
WORKDIR /workspace/hover_net
RUN pip install -r requirements.txt

WORKDIR /mnt/dataset
RUN git clone https://github.com/suren3141/MoNuSegDataset.git MoNuSeg

WORKDIR /workspace
RUN git clone https://github.com/suren3141/hover_net.git
WORKDIR /workspace/semantic-diffusion-model

# RUN pip install -r environment/requirements.txt

