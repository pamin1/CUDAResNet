#include "Kernel.cuh"

__global__ void conv2d_kernel(const float *input,                // [in_channels, H, W]
                              const float *weight,               // [out_channels, in_channels, kH, kW]
                              const float *bnWeight,             //
                              const float *bnBias,               //
                              const float *bnMean,               //
                              const float *bnVar,                //
                              float *output,                     // [out_channels, H_out, W_out]
                              int in_channels, int out_channels, // channels
                              int H, int W, int outH, int outW,  // dims
                              int kernel_size, int stride, int padding)
{
    int oc = blockIdx.z;
    int out_h = blockIdx.y * blockDim.y + threadIdx.y;
    int out_w = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_h >= outH || out_w >= outW)
    {
        return;
    }

    float sum = 0.0f;
    for (int ic = 0; ic < in_channels; ic++)
    {
        for (int kh = 0; kh < kernel_size; kh++)
        {
            for (int kw = 0; kw < kernel_size; kw++)
            {
                int in_h = out_h * stride + kh - padding;
                int in_w = out_w * stride + kw - padding;

                // Check bounds (padding)
                if (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W)
                {
                    int input_idx = ic * H * W + in_h * W + in_w;
                    int weight_idx = oc * in_channels * kernel_size * kernel_size +
                                     ic * kernel_size * kernel_size + kh * kernel_size + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    int output_idx = oc * outH * outW + out_h * outW + out_w;
    float scale = bnWeight[oc] / sqrtf(bnVar[oc] + EPSILON);

    output[output_idx] = scale * (sum - bnMean[oc]) + bnBias[oc];
}

__global__ void downsample_kernel(const float *input,     // [in_ch, H, W]
                                  const float *weight,    // [out_ch, in_ch, 1, 1]
                                  const float *bn_weight, //
                                  const float *bn_bias,   //
                                  const float *bn_mean,   //
                                  const float *bn_var,    //
                                  float *output,          // [out_ch, H/2, W/2]
                                  int in_ch, int out_ch, int H, int W, float epsilon)
{
    int oc = blockIdx.z;
    int out_h = blockIdx.y * blockDim.y + threadIdx.y;
    int out_w = blockIdx.x * blockDim.x + threadIdx.x;

    // stride 2
    int outH = (H + 1) / 2;
    int outW = (W + 1) / 2;

    if (out_h >= outH || out_w >= outW)
        return;

    // position also strided
    int in_h = out_h * 2;
    int in_w = out_w * 2;

    // 1×1 convolution
    float sum = 0.0f;
    for (int ic = 0; ic < in_ch; ic++)
    {
        int input_idx = ic * H * W + in_h * W + in_w;
        int weight_idx = oc * in_ch + ic; // 1×1 kernel
        sum += input[input_idx] * weight[weight_idx];
    }

    // inline bn
    float scale = bn_weight[oc] / sqrtf(bn_var[oc] + epsilon);
    float normalized = scale * (sum - bn_mean[oc]) + bn_bias[oc];

    int output_idx = oc * outH * outW + out_h * outW + out_w;
    output[output_idx] = normalized;
}

__global__ void relu_kernel(const float *input, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void add_kernel(const float *a, const float *b, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output[idx] = a[idx] + b[idx];
    }
}

__global__ void maxpool_kernel(const float *input, // [C, H, W]
                               float *output,      // [C, H_out, W_out]
                               int C, int H, int W, int kernel_size, int stride, int padding)
{
    int c = blockIdx.z;
    int out_h = blockIdx.y * blockDim.y + threadIdx.y;
    int out_w = blockIdx.x * blockDim.x + threadIdx.x;

    int outH = (H + 2 * padding - kernel_size) / stride + 1;
    int outW = (W + 2 * padding - kernel_size) / stride + 1;

    if (out_h >= outH || out_w >= outW)
        return;

    float max_val = -INFINITY;

    for (int kh = 0; kh < kernel_size; kh++)
    {
        for (int kw = 0; kw < kernel_size; kw++)
        {
            int in_h = out_h * stride + kh - padding;
            int in_w = out_w * stride + kw - padding;

            if (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W)
            {
                int input_idx = c * H * W + in_h * W + in_w;
                max_val = fmaxf(max_val, input[input_idx]);
            }
        }
    }

    int output_idx = c * outH * outW + out_h * outW + out_w;
    output[output_idx] = max_val;
}

__global__ void adaptive_avgpool_kernel(const float *input, // [C, H, W]
                                        float *output,      // [C, 1, 1]
                                        int C, int H, int W)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (c >= C)
        return;

    float sum = 0.0f;
    int size = H * W;

    for (int i = 0; i < size; i++)
    {
        sum += input[c * size + i];
    }

    output[c] = sum / size;
}

__global__ void fc_kernel(const float *input,  // [in_features]
                          const float *weight, // [out_features, in_features]
                          const float *bias,   // [out_features]
                          float *output,       // [out_features]
                          int in_features, int out_features)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_idx >= out_features)
        return;

    float sum = 0.0f;
    for (int i = 0; i < in_features; i++)
    {
        int weight_idx = out_idx * in_features + i;
        sum += input[i] * weight[weight_idx];
    }

    output[out_idx] = sum + bias[out_idx];
}

void launchMaxPoolKernel(float *input, float *output, int H, int W, int C, int kernel_size,
                         int stride, int padding)
{
    int outH = (H + 2 * padding - kernel_size) / stride + 1;
    int outW = (W + 2 * padding - kernel_size) / stride + 1;

    dim3 block(16, 16, 1);
    dim3 grid((outW + 15) / 16, (outH + 15) / 16, C);

    maxpool_kernel<<<grid, block>>>(input, output, C, H, W, kernel_size, stride, padding);
}

void launchConvKernel(float *image, float *output, const ConvLayer &conv, const BatchNorm &bn,
                      int inputDim, int stride, int pad)
{
    int outputH = computeDim(inputDim, stride, pad, conv.kernelSize);
    int outputW = computeDim(inputDim, stride, pad, conv.kernelSize);

    dim3 block(16, 16, 1);
    dim3 grid((outputW + block.x - 1) / block.x, (outputH + block.y - 1) / block.y, conv.outputSize);

    conv2d_kernel<<<grid, block>>>(image,                                               //
                                   conv.d_weight,                                       //
                                   bn.d_weight,                                         //
                                   bn.d_bias,                                           //
                                   bn.d_runningMean,                                    //
                                   bn.d_runningVar,                                     //
                                   output,                                              //
                                   conv.inputSize, conv.outputSize, inputDim, inputDim, //
                                   outputH, outputW,                                    //
                                   conv.kernelSize, stride, pad);
}

void launchDownsampleKernel(float *input, float *output, const Downsample &ds, int H, int W)
{
    int outH = (H + 1) / 2;
    int outW = (W + 1) / 2;

    dim3 block(16, 16, 1);
    dim3 grid((outW + 15) / 16, (outH + 15) / 16, ds.weight.outputSize);

    downsample_kernel<<<grid, block>>>(input,               //
                                       ds.weight.d_weight,  //
                                       ds.bn.d_weight,      //
                                       ds.bn.d_bias,        //
                                       ds.bn.d_runningMean, //
                                       ds.bn.d_runningVar,  //
                                       output,              //
                                       ds.weight.inputSize, ds.weight.outputSize, H, W,
                                       1e-5f // epsilon
    );
}

void launchAddKernel(float *a, float *b, float *output, int size)
{
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    add_kernel<<<gridSize, blockSize>>>(a, b, output, size);
}

void launchReLUKernel(float *input, float *output, int size)
{
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    relu_kernel<<<gridSize, blockSize>>>(input, output, size);
}

void runBasicBlock(const BasicBlock &bb, float *input, float *output, int inputChannels,
                   int inputH, int inputW,
                   int stride1) // stride for first conv (1 or 2)
{
    // Determine if this is a downsampling block
    bool isDownsample = (stride1 == 2);

    // Calculate intermediate dimensions
    int midH = (inputH + 2 * 1 - 3) / stride1 + 1; // After conv1
    int midW = (inputW + 2 * 1 - 3) / stride1 + 1;
    int outH = midH; // After conv2 (stride=1)
    int outW = midW;

    // Allocate temporary buffers
    float *temp1, *temp2, *identity;
    size_t temp1_size = bb.conv1.outputSize * midH * midW * sizeof(float);
    size_t temp2_size = bb.conv2.outputSize * outH * outW * sizeof(float);

    cudaMalloc(&temp1, temp1_size);
    cudaMalloc(&temp2, temp2_size);
    cudaMalloc(&identity, temp2_size); // Same size as final output

    // Main path
    // Conv1 + BN1 + ReLU
    launchConvKernel(input, temp1, bb.conv1, bb.bn1, inputH, stride1, 1); // padding=1 for 3x3
    launchReLUKernel(temp1, temp1, bb.conv1.outputSize * midH * midW);

    // Conv2 + BN2 (no ReLU yet)
    launchConvKernel(temp1, temp2, bb.conv2, bb.bn2, midH, 1, 1); // stride=1, padding=1

    // Identity/Skip path
    if (bb.hasDownsample)
    {
        // Downsample: 1x1 conv with stride=2
        launchDownsampleKernel(input, identity, bb.ds, inputH, inputW);
    }
    else
    {
        // Direct copy (dimensions match)
        size_t copy_size = inputChannels * inputH * inputW * sizeof(float);
        cudaMemcpy(identity, input, copy_size, cudaMemcpyDeviceToDevice);
    }

    // Add: output = temp2 + identity
    launchAddKernel(temp2, identity, output, bb.conv2.outputSize * outH * outW);

    // Final ReLU
    launchReLUKernel(output, output, bb.conv2.outputSize * outH * outW);

    // Cleanup
    cudaFree(temp1);
    cudaFree(temp2);
    cudaFree(identity);
}

void launchAdaptiveAvgPoolKernel(float *input, float *output, int H, int W, int C)
{
    int blockSize = 256;
    int gridSize = (C + blockSize - 1) / blockSize;

    adaptive_avgpool_kernel<<<gridSize, blockSize>>>(input, output, C, H, W);
}

void launchFCKernel(float *input, float *output, const FullyConnected &fc, int in_features,
                    int out_features)
{
    int blockSize = 256;
    int gridSize = (out_features + blockSize - 1) / blockSize;

    fc_kernel<<<gridSize, blockSize>>>(input, fc.d_weight, fc.d_bias, output, in_features,
                                       out_features);
}