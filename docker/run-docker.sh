#!/bin/bash
RUN_DIR=$(dirname $(readlink -f $0))
PKG_DIR=$(dirname ${RUN_DIR})

docker run \
    -it \
    --rm \
    -v ${PKG_DIR}:/workspace:rw \
    --name pointsmap-python \
    pointsmap-python:build \
    bash
