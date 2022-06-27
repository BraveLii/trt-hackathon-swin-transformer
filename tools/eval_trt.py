import os
import sys
import ctypes
import numpy as np
import argparse
import time
import tensorrt as trt
from cuda import cudart
import pycuda.autoinit
import pycuda.driver as cuda

parser = argparse.ArgumentParser('Swin Transformer evaluation script', add_help=True)
parser.add_argument('--model', type=str, default="swinv2_base_patch4_window16_256.plan", help='run pytorch model')
parser.add_argument('--baseline', type=str, default="torch.npy", help='baseline file')
args, unparsed = parser.parse_known_args()

soFile = "/workspace/plugin/layerNorm/LayerNormPlugin.so"
input_data = np.ones((1,3,256,256)).astype(np.float32)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
ctypes.cdll.LoadLibrary(soFile)
with open(args.model, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

with engine.create_execution_context() as context:
    h_input = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float32)
    h_input = h_input.reshape(context.get_binding_shape(0))
    h_input = np.array(h_input)
    h_input += input_data
    h_output = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float32)
    h_output = np.array(h_output)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    bufferD = []
    bufferD.append(d_input)
    bufferD.append(d_output)
    cuda.memcpy_htod(d_input, h_input)
    context.execute_v2(bufferD)
    cuda.memcpy_dtoh(h_output, d_output)

    for i in range(10):
        context.execute_v2(bufferD)

    start = time.time()
    for i in range(100):
        context.execute_v2(bufferD)
    end = time.time()
    cost = (end-start)*1000

    output_trt = h_output.reshape(1,1000)
    print("trt output:")
    print(output_trt)

print("=== trt infer ===")
print("model path: {}".format(args.model))
print("infer 100 cost: {:.2f} ms".format(cost))
print("average time: {:.2f} ms".format(cost/100.0))
print("fps: {:.2f}".format(100.0/cost*1000))


sys.path.append("/workspace/scripts")
from utils import check_diff

baseline = np.load(args.baseline)
check10, check11, check12 = check_diff(baseline, output_trt, True, 5e-5)

print("=== trt accuracy ===")
print("max abs diff: {:.2E}".format(check11))
print("median rel diff: {:.2E}".format(check12))