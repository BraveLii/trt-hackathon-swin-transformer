#命令：python mymain.py --cfg configs/swinv2/swinv2_base_patch4_window16_256.yaml

import os
import argparse
import torch
import sys
sys.path.append("./Swin-Transformer/")
import config
from models import build_model

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, default="Swin-Transformer/configs/swinv2/swinv2_base_patch4_window16_256.yaml", metavar="FILE", help='path to config file', )
    args, unparsed = parser.parse_known_args()

    conf = config._C.clone()
    config._update_config_from_file(conf, args.cfg)
    conf.defrost()
    conf.freeze()

    return conf, args

@torch.no_grad()
def validate(model, onnx_model):  
    # device = torch.device("cpu")
    # model.to(device)
    model.eval()
    
    dummy_input = torch.randn(10, 3, 256, 256, device="cpu")
    model(dummy_input)
    print("******* after model run ********")
    # os._exit(0)
    input_names = [ "actual_input_1" ]
    output_names = [ "output1" ]
    torch.onnx.export(model, dummy_input, onnx_model, verbose=True, input_names=input_names, output_names=output_names, opset_version=11)
    
def main(config, args):
    model = build_model(config)
    model_name = args.cfg.split(".")[0].split("/")[-1]
    print("model name : ", model_name)
    pth_model = "{}.pth".format(model_name)
    onnx_model = "{}.onnx".format(model_name)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        print(f"number of GFLOPs: {flops / 1e9}")

    model.load_state_dict(torch.load(pth_model)['model'], strict=True)
    
    validate(model, onnx_model)

if __name__ == '__main__':
    config, args = parse_option()
    main(config, args)