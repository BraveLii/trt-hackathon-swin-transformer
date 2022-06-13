# build model

## 下载代码
    git clone https://github.com/BraveLii/trt-hackathon-swin-transformer.git
    cd trt-hackathon-swin-transformer
    git submodule update --init

## 进入docker并安装依赖
    ./start_torch.sh
    ./install.sh

## 模型导出和转换
    预训练模型下载地址: https://github.com/SwinTransformer/storage/releases/tag/v2.0.0
    以swinv2_base_patch4_window16_256模型为例：
    python tools/swinv2_onnx.py --cfg Swin-Transformer/configs/swinv2/swinv2_base_patch4_window16_256.yaml
    ./tools/build_trt.sh
    
