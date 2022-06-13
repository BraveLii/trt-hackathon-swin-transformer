#!/bin/bash

docker run --runtime=nvidia -ti  -v $(pwd):/workspace -w /workspace  $@ --privileged -v /var/run/docker.sock:/var/run/docker.sock nvcr.io/nvidia/pytorch:21.12-py3
