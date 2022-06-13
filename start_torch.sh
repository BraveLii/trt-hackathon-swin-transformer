#!/bin/bash
image="nvcr.io/nvidia/pytorch:21.12-py3"
nvidia-docker run  -ti  -v $(pwd):/workspace -w /workspace  $@ --privileged -v /var/run/docker.sock:/var/run/docker.sock $image
