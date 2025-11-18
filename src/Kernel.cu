#include "Kernel.cuh"

__global__ void conv2d_kernel(const float *input,  // [in_channels, H, W]
                              const float *weight, // [out_channels, in_channels, kH, kW]
                              float *output,       // [out_channels, H_out, W_out]
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
  output[output_idx] = sum;
}

void launchConvKernel(float *image, float *output, const ConvLayerDev &conv, int inputDim,
                      int stride, int pad)
{
  int outputH = computeDim(inputDim, stride, pad, conv.kernelSize);
  int outputW = computeDim(inputDim, stride, pad, conv.kernelSize);

  std::cout << "oH: " << outputH   //
            << "\noW: " << outputW //
            << "\noS: " << conv.outputSize << "\n";

  dim3 block(16, 16, 1);
  dim3 grid((outputW + block.x - 1) / block.x, // Fixed: use block.x, add parentheses
            (outputH + block.y - 1) / block.y, // Fixed: use block.y, add parentheses
            conv.outputSize);

  conv2d_kernel<<<grid, block>>>(image, conv.d_weight, output, conv.inputSize, conv.outputSize,
                                 inputDim, inputDim, // input height, width
                                 outputH, outputW,   // Fixed: use outputH, outputW
                                 conv.kernelSize, stride, pad);
}