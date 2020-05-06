#!/bin/bash

docker run \
    --rm \
    -p 127.0.0.1:8888:8888 \
    -p 127.0.0.1:6006:6006 \
    -v $(pwd):/home/jovyan/work \
    -v $(pwd)/tf-logs:/home/jovyan/tf-logs \
    -v $(pwd)/keras-cache:/home/jovyan/.keras \
    tf
