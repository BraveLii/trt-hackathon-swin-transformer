/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 
 #include "LayerNormPlugin.h"
 #include <iostream>

using namespace nvinfer1;

PluginFieldCollection LayerNormPluginCreator::fc_{};
std::vector<PluginField> LayerNormPluginCreator::attr_;

template <int N> 
__global__ void layerNormKernel(float *pInput, float *pOutput)
{
    const int tx = threadIdx.x, index = blockIdx.x * N + threadIdx.x;

    __shared__ float temp[N/2];

    float value0 = pInput[index];
    float value1 = pInput[index + N/2];

    temp[tx] = value0 + value1;
    __syncthreads();

    for (int stride = N/4; stride >= 1; stride /= 2)
    {
        if (tx < stride)
        {
            temp[tx] += temp[tx + stride];
        }
        __syncthreads();
    }
    float mean = temp[0] / N;
    __syncthreads();

    temp[tx] = (value0 - mean) * (value0 - mean) + (value1 - mean) * (value1 - mean);
    __syncthreads();

    for (int stride = N/4; stride >= 1; stride /= 2)
    {
        if (tx < stride)
        {
            temp[tx] += temp[tx + stride];
        }
        __syncthreads();
    }
    float var = temp[0] / N;

    pOutput[index]       = (value0 - mean) * rsqrtf(var + 1e-5);
    pOutput[index + N/2] = (value1 - mean) * rsqrtf(var + 1e-5);
}

int32_t LayerNormPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const int nBlock = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];
    const int nValuePerBlock = inputDesc[0].dims.d[inputDesc[0].dims.nbDims-1];

    std::cout << "nbDims: " << inputDesc[0].dims.nbDims << std::endl;
    std::cout << "nBlock: " << nBlock << std::endl;
    std::cout << "nValuePerBlock: " << nValuePerBlock << std::endl;
    // std::cout << "inputDesc[0].dims.d[2]: " << inputDesc[0].dims.d[2] << std::endl;
    // std::cout << "inputDesc[0].dims.d[3]: " << inputDesc[0].dims.d[3] << std::endl;

    switch(nValuePerBlock){
        case 128:
            layerNormKernel<128><<<nBlock, nValuePerBlock/2, 0, stream>>>((float *)inputs[0], (float *)outputs[0]);
            break;
        case 256:
            layerNormKernel<256><<<nBlock, nValuePerBlock/2, 0, stream>>>((float *)inputs[0], (float *)outputs[0]);
            break;
        case 512:
            layerNormKernel<512><<<nBlock, nValuePerBlock/2, 0, stream>>>((float *)inputs[0], (float *)outputs[0]);
            break;
        case 1024:
            layerNormKernel<1024><<<nBlock, nValuePerBlock/2, 0, stream>>>((float *)inputs[0], (float *)outputs[0]);
            break;
        default:
            std::cout << "LayerNorm plugin does not support nValuePerBlock = " << nValuePerBlock << std::endl;
            return -1;
    }

    std::cout << "end nBlock: " << nBlock << std::endl;
    std::cout << "end nValuePerBlock: " << nValuePerBlock << std::endl;
    
    return 0;
}

REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);

