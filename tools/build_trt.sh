#!/bin/bash
trtexec  --workspace=8192  \
--onnx=swin_small_patch4_window7_224.onnx \
--saveEngine=swin_small_patch4_window7_224.plan \
--verbose
