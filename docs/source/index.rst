================
pointsmap-python
================

Python3で点群地図を扱うライブラリ.

依存
====

* Linux
* CMake
* Ninja-Build
* Python >=3.6
* Numpy
* OpenCV >=3.2
* Point Cloud Library >=1.8

インストール
============

Docker
------

次のコマンドを実行して, 既存のDockerイメージにpointsmap-pythonをインストールする.

.. code-block:: bash

    git clone https://github.com/shikishima-TasakiLab/pointsmap-python.git
    cd pointsmap-python
    ./installer/install-env-docker.sh -i BASE_IMAGE[:TAG]

Local
-----

.. code-block:: bash

    apt update
    apt install \
        build-essential \
        git \
        python3-pip \
        python3-dev \
        python3-numpy-dev \
        libboost-python-dev \
        libboost-numpy-dev \
        libpcl-dev \
        libopencv-dev
    pip install scikit-build cmake ninja
    pip install git+https://github.com/shikishima-TasakiLab/pointsmap-python

使い方
======

.. toctree::
    :maxdepth: 2
    :glob:

    *
