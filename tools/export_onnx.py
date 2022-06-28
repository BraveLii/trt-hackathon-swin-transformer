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
    parser.add_argument('--cfg', type=str, default="Swin-Transformer/configs/swinv2/swinv2_base_patch4_window16_256.yaml", metavar="FILE", help='path to config file')
    parser.add_argument('--fold-constant', action="store_true", help='whether to fold constant when exporting onnx')
    args = parser.parse_args()

    conf = config._C.clone()
    config._update_config_from_file(conf, args.cfg)
    conf.defrost()
    conf.freeze()

    return conf, args

@torch.no_grad()
def validate(args, model, onnx_model):  
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 256, 256, device="cpu")
    # model(dummy_input)

    print("******* after model run ********")
    # os._exit(0)
    input_names = [ "actual_input_1" ]
    output_names = [ "output1" ]
    print("args.fold_constant: ", args.fold_constant)
    torch.onnx.export(model, dummy_input, onnx_model, verbose=True, input_names=input_names, output_names=output_names, opset_version=11, do_constant_folding=True)
    
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

    checkpoint = torch.load(pth_model,  map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'], strict=False)
    
    validate(args, model, onnx_model)

if __name__ == '__main__':
    config, args = parse_option()
    main(config, args)