#!/bin/bash
pip install timm==0.4.12 opencv-python opencv-contrib-python termcolor==1.1.0 yacs==0.1.8 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install --user --upgrade git+https://github.com/microsoft/tutel@main
apt-get update && apt-get install -y libgl1 