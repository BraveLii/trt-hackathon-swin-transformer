#!/bin/bash
image="registry.cn-hangzhou.aliyuncs.com/trt2022/trt-8.4-ga:latest"
nvidia-docker run  -ti  -v $(pwd):/workspace -w /workspace  $@ --privileged -v /var/run/docker.sock:/var/run/docker.sock $image
