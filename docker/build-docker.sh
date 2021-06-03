#!/bin/bash
BUILD_DIR=$(dirname $(readlink -f $0))

docker build \
    -t pointsmap-python:build \
    ${BUILD_DIR}