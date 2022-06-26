import onnx_graphsurgeon as gs
import numpy as np
import onnx

scale_const = gs.Constant(name="scale_const_1", values=np.ones(shape=(63,), dtype=np.float32))
bias_const = gs.Constant(name="b_const_1", values=np.zeros(shape=(63,), dtype=np.float32))

@gs.Graph.register()
def replace_with_instancenormalization(self, inputs, outputs):
    print("inputs : ", inputs)
    print("outputs : ", outputs)

    # Disconnect output nodes of all input tensors
    for inp in inputs:
        inp.outputs = [o for o in inp.outputs if not o.name.startswith("ReduceMean_") and not o.name.startswith("Sub_")]
        # inp.outputs.clear()

    # # Disconnet input nodes of all output tensors
    for out in outputs:
        out.inputs.clear()

    # Insert the new node.
    print("inputs[0].shape: ", inputs[0])

    node = self.layer(op="LayerNorm", inputs=[inputs[0]], outputs=outputs, attrs={"epsilon": 0.00001})
    # node = self.layer(op="InstanceNormalization", inputs=[inputs[0], scale_const, bias_const], outputs=outputs)
    print("node: ", node)
    return node


def merge_instance_norm(graph):
    reducemean_nodes = [node for node in graph.nodes if node.name.startswith("ReduceMean") and node.outputs[0].outputs[0].name.startswith("Sub_")]
    div_nodes = [node for node in graph.nodes if node.name.startswith("Div_") and node.inputs[0].inputs[0].name.startswith("Sub_") and node.inputs[1].inputs[0].name.startswith("Sqrt")]
    mul_nodes = [ node.outputs[0].outputs[0] for node in div_nodes]
    add_nodes = [ node.outputs[0].outputs[0] for node in mul_nodes]

    print("len(reducemean_nodes): ", len(reducemean_nodes))
    print("len(div_nodes): ", len(div_nodes))
    print("len(mul_nodes): ", len(mul_nodes))
    print("len(add_nodes): ", len(add_nodes))
    assert len(reducemean_nodes) == len(div_nodes), "reducemean_nodes len is not equal with div_nodes len"

    for rn, dn in zip(reducemean_nodes, div_nodes):
        inputs = [inp for inp in rn.inputs]
        outputs = [out for out in dn.outputs]
        # scale = mn.inputs[1]
        # bias = an.inputs[1]

        graph.replace_with_instancenormalization(inputs, outputs)