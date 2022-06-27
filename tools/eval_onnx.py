import os
import sys
import numpy as np
import argparse
import time
import onnxruntime

parser = argparse.ArgumentParser('Swin Transformer evaluation script', add_help=True)
parser.add_argument('--model', type=str, default="swinv2_base_patch4_window16_256.onnx", help='run pytorch model')
parser.add_argument('--baseline', type=str, default="torch.npy", help='baseline file')
args, unparsed = parser.parse_known_args()

input_data = np.ones((1,3,256,256)).astype(np.float32)

session = onnxruntime.InferenceSession(args.model, providers=["CUDAExecutionProvider"])

for i in range(20):
    outputs = session.run(['output1'], {"actual_input_1": input_data})


start = time.time()
for i in range(100):
    outputs = session.run(['output1'], {"actual_input_1": input_data})
end = time.time()
cost = (end-start)*1000

output_onnx = np.array(outputs[0])

print("onnx output:")
print(output_onnx)

print("=== onnx infer ===")
print("model path: {}".format(args.model))
print("infer 100 cost: {:.2f} ms".format(cost))
print("average time: {:.2f} ms".format(cost/100.0))
print("fps: {:.2f}".format(100.0/cost*1000))


sys.path.append("/workspace/scripts")
from utils import check_diff

baseline = np.load(args.baseline)
check10, check11, check12 = check_diff(baseline, output_onnx, True, 5e-5)

print("=== onnx accuracy ===")
print("max abs diff: {:.2E}".format(check11))
print("median rel diff: {:.2E}".format(check12))