import onnx_graphsurgeon as gs
import numpy as np
import onnx
import argparse

import sys
sys.path.append("/workspace/scripts")
import merge

parser = argparse.ArgumentParser('Swin Transformer evaluation script', add_help=True)
parser.add_argument('--model', type=str, default="swinv2_base_patch4_window16_256.onnx", help='run pytorch model')
args = parser.parse_args()


onnx_model = args.model
opt_model = onnx_model.split(".")[0]+"_opt.onnx"

print("onnx model: ", onnx_model)
print("onoptnx model: ", opt_model)

graph = gs.import_onnx(onnx.load(onnx_model))
tmap = graph.tensors()


# print(tmap["478"].inputs[0])
# print(tmap["484"].inputs[0])

merge.merge_instance_norm(graph)

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), opt_model)
