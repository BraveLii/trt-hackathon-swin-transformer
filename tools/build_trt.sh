#!/bin/bash
if [ $# -lt "2" ]
then
    echo "run like: ./tools/build_trt.sh swinv2_base_patch4_window16_256.onnx swinv2_base_patch4_window16_256.plan"
fi

onnx_model=$1
trt_model=$2

echo "building from $onnx_model ---> $trt_model"

trtexec  --workspace=8192  \
--onnx=$onnx_model \
--saveEngine=$trt_model \
--verbose
