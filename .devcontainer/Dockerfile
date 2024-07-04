FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV TZ=Europe/Berlin
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt -y install git curl bc libgl1-mesa-glx libglib2.0-0 htop msttcorefonts

# Install miniconda
RUN cd /root && curl -O -J https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
ENV PATH="${PATH}:/root/miniconda3/bin"
RUN conda init bash

# Create working directory
RUN mkdir -p /root/industrial-clip
WORKDIR /root/industrial-clip

# Create conda environment
RUN conda create -y -n industrial-clip python=3.8
RUN echo "conda activate industrial-clip" >> $HOME/.bashrc
ENV CONDA_DEFAULT_ENV industrial-clip
SHELL ["conda", "run", "-n", "industrial-clip", "/bin/bash", "-c"]

# Install torch
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installation
# Install dassl
RUN cd /root/ && git clone https://github.com/KaiyangZhou/Dassl.pytorch.git /root/dassl
RUN cd /root/dassl && pip install -r requirements.txt
RUN cd /root/dassl && python setup.py develop
RUN curl -L https://gist.github.com/kenomo/c65749eb1817874c2f348890cac6cde6/raw/5f4bc86158c365024b00df17beeb7bc422455f5f/optimizer.py > /root/dassl/dassl/optim/optimizer.py

# Post-installation
RUN pip install setuptools==59.5.0
RUN pip install open_clip_torch
RUN pip install npy-append-array
RUN pip install scikit-learn
RUN pip install orjson
RUN pip install wandb
RUN pip install linetimer
RUN pip install git+https://github.com/facebookresearch/segment-anything.git
RUN pip install opencv-python matplotlib
RUN pip install jupyter supervision
