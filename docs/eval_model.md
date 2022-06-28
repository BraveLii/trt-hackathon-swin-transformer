# eval model

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
