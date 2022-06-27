#!/bin/bash
set -e

if [ $# -lt "2" ]
then 
    echo "run like: ./tools/build_trt.sh swinv2_base_patch4_window16_256.onnx swinv2_base_patch4_window16_256.plan"
    exit 0
fi

echo "build plugin"
pushd /workspace/plugin/layerNorm
make clean && make -j4
popd

onnx_model=$1
trt_model=$2

echo "building from $onnx_model ---> $trt_model"


trtexec  --workspace=1073741824  \
--onnx=$onnx_model \
--saveEngine=$trt_model \
--plugins=/workspace/plugin/layerNorm/LayerNormPlugin.so \
--dumpOutput \
# --verbose
