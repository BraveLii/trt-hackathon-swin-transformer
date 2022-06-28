# eval model

## 环境依赖
+ 主机硬件环境：Linux version 5.4.0-110-generic (buildd@ubuntu) (gcc version 9.4.0 (Ubuntu 9.4.0-1ubuntu1~20.04.1)) , NVIDIA显卡： NVIDIA A10 (24G)

+ 主机软件环境：Driver Version: 510.73.08, CUDA 11.6/cuDNN8.4/TRT8.4, Docker和NVIDIA-Docker

+ Docker镜像的使用具体可以参考：https://github.com/NVIDIA/trt-samples-for-hackathon-cn/blob/master/hackathon/setup.md

## 下载代码
    git clone https://github.com/BraveLii/trt-hackathon-swin-transformer.git
    cd trt-hackathon-swin-transformer
    git submodule update --init

## eval torch模型
    ./start_torch.sh
    ./install.sh
    使用下面脚本运行torch模型，并产生baseline文件：
    > python tools/eval_torch.py    

## eval onnx模型
    ./start_trt.sh
    使用下面脚本运行onnx模型，该过程依赖baseline文件：
     python tools/eval_onnx.py  

## eval trt模型
    ./start_trt.sh
    使用下面脚本运行trt模型，该过程依赖baseline文件：
    python tools/eval_trt.py 


## optimize onnx模型
    ./start_trt.sh
    python tools/opt_onnx.py 
