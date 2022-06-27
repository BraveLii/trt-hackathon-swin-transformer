import os
import sys
import time
import numpy as np
import torch
import argparse
import sys
sys.path.append("./Swin-Transformer/")
import config
from models import build_model

parser = argparse.ArgumentParser('Swin Transformer evaluation script', add_help=True)
parser.add_argument('--model', type=str, default="swinv2_base_patch4_window16_256.pth", help='run pytorch model')
parser.add_argument('--baseline', type=str, default="torch.npy", help='baseline file')
args, unparsed = parser.parse_known_args()

conf = config._C.clone()
config._update_config_from_file(conf, 'Swin-Transformer/configs/swinv2/swinv2_base_patch4_window16_256.yaml')

model = build_model(conf)
model.cuda()

checkpoint = torch.load(args.model, map_location='cuda')
model.load_state_dict(checkpoint['model'], strict=True)

model.eval()

input_data = np.ones((1,3,256,256)).astype(np.float32)

with torch.no_grad():
    for i in range(10):
        output = model(torch.tensor(input_data).cuda())
    
    start = time.time()
    for i in range(100):
        output = model(torch.tensor(input_data).cuda())
    end = time.time()
    cost = (end-start)*1000

output_data = output.cpu().numpy()
print(output_data)

print("=== torch result ===")
print("infer 100 cost: {:.2f} ms".format(cost))
print("average time: {:.2f} ms".format(cost/100.0))
print("fps: {:.2f}".format(100.0/cost*1000))


np.save('torch', output_data)
print("save torch result to torch.npy")
