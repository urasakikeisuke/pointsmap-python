FROM ubuntu:18.04

RUN sed -i.bak -e "s%http://[^ ]\+%http://ftp.jaist.ac.jp/pub/Linux/ubuntu/%g" /etc/apt/sources.list

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        git \
        cmake \
        python3-dev \
        libopencv-dev \
        python3-opencv \
        python3-h5py \
        libpcl-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y \
        curl \
        ninja-build && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y \
        python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install scikit-build cmake ninja twine

WORKDIR /workspace