import onnx_graphsurgeon as gs
import numpy as np
import onnx

import sys
sys.path.append("/workspace/scripts")
import merge

onnx_model = "/workspace/swinv2_base_patch4_window16_256.onnx"
modified_model = "/workspace/swinv2_base_patch4_window16_256_opt.onnx"

graph = gs.import_onnx(onnx.load(onnx_model))
tmap = graph.tensors()


# print(tmap["478"].inputs[0])
# print(tmap["484"].inputs[0])

merge.merge_instance_norm(graph)

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), modified_model)
