ARG SRC_IMAGE
FROM ${SRC_IMAGE}

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        libboost-python* \
        libboost-numpy* \
        libpcl-common* \
        libpcl-io* \
        libpcl-filters* \
        libpcl-segmentation* \
        libpcl-recognition* \
        libopencv-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install scikit-build cmake ninja  && \
    pip3 install git+https://github.com/shikishima-TasakiLab/pointsmap-python
