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
    python tools/export_onnx.py --cfg Swin-Transformer/configs/swinv2/swinv2_base_patch4_window16_256.yaml

    进入trt镜像
    ./start_trt.sh

    如果需要对onnx模型进行优化，否则跳过此步骤(目前只有合并LayerNorm,合并之后速度变慢，且结果不正确，暂未解决)
    python tools/opt_onnx.py 

    build trt模型需要升级trt到8.4.0版本，否则会报错
    ./tools/build_trt.sh swinv2_base_patch4_window16_256.onnx swinv2_base_patch4_window16_256.plan
    
