#include "CopyModel.h"
#include "cuda_runtime.h"

void CopyModel::copyConvLayer(ConvLayerDev &dst, const ConvLayer &src)
{
  dst.outputSize = src.outputSize;
  dst.inputSize = src.inputSize;
  dst.kernelSize = src.kernelSize;

  // allocate and copy
  size_t size = dst.outputSize * dst.inputSize * dst.kernelSize * dst.kernelSize;
  CHECK_ERROR(cudaMalloc((void **)&dst.d_weight, sizeof(float) * size));
  CHECK_ERROR(cudaMemcpy((void *)dst.d_weight, (const void *)src.h_weight, size * sizeof(float), cudaMemcpyHostToDevice));
}

void CopyModel::copyBatchNorm(BatchNormDev &dst, const BatchNorm &src)
{
  dst.numFeatures = src.numFeatures;
  size_t size = sizeof(float) * dst.numFeatures;

  // allocate
  CHECK_ERROR(cudaMalloc((void **)&dst.d_weight, size));
  CHECK_ERROR(cudaMalloc((void **)&dst.d_bias, size));
  CHECK_ERROR(cudaMalloc((void **)&dst.d_runningMean, size));
  CHECK_ERROR(cudaMalloc((void **)&dst.d_runningVar, size));

  // copy
  CHECK_ERROR(cudaMemcpy((void *)dst.d_weight, (const void *)src.h_weight, size, cudaMemcpyHostToDevice));
  CHECK_ERROR(cudaMemcpy((void *)dst.d_bias, (const void *)src.h_bias, size, cudaMemcpyHostToDevice));
  CHECK_ERROR(cudaMemcpy((void *)dst.d_runningMean, (const void *)src.h_runningMean, size, cudaMemcpyHostToDevice));
  CHECK_ERROR(cudaMemcpy((void *)dst.d_runningVar, (const void *)src.h_runningVar, size, cudaMemcpyHostToDevice));
}

void CopyModel::copyFullyConnected(FullyConnectedDev &dst, const FullyConnected &src)
{
  dst.outputSize = src.outputSize;
  dst.inputSize = src.inputSize;
  size_t wSize = sizeof(float) * dst.outputSize * dst.inputSize;
  size_t bSize = sizeof(float) * dst.outputSize;

  // allocate
  CHECK_ERROR(cudaMalloc((void **)&dst.d_weight, wSize));
  CHECK_ERROR(cudaMalloc((void **)&dst.d_bias, bSize));

  // copy
  CHECK_ERROR(cudaMemcpy((void *)dst.d_weight, (const void *)src.h_weight, wSize, cudaMemcpyHostToDevice));
  CHECK_ERROR(cudaMemcpy((void *)dst.d_bias, (const void *)src.h_bias, bSize, cudaMemcpyHostToDevice));
}

void CopyModel::copyDownSample(DownsampleDev &dst, const Downsample &src)
{
  copyConvLayer(dst.weight, src.weight);
  copyBatchNorm(dst.bn, src.bn);
}

void CopyModel::copyBasicBlock(BasicBlockDev &dst, const BasicBlock &src)
{
  copyConvLayer(dst.conv1, src.conv1);
  copyConvLayer(dst.conv2, src.conv2);

  copyBatchNorm(dst.bn1, src.bn1);
  copyBatchNorm(dst.bn2, src.bn2);

  dst.hasDownsample = src.hasDownsample;
  copyDownSample(dst.ds, src.ds);
}

CopyModel::CopyModel(const ResNet18 &model)
{
  copyConvLayer(devModel.conv1, model.conv1);
  // copyBatchNorm(devModel.bn1, model.bn1);

  // auto copyLayers = [&](BasicBlockDev *dst, const BasicBlock *src)
  // {
  //   copyBasicBlock(dst[0], src[0]);
  //   copyBasicBlock(dst[1], src[1]);
  // };

  // copyLayers(devModel.layer1, model.layer1);
  // copyLayers(devModel.layer2, model.layer2);
  // copyLayers(devModel.layer3, model.layer3);
  // copyLayers(devModel.layer4, model.layer4);

  // copyFullyConnected(devModel.fc, model.fc);

  std::cout << "Successfully loaded model on GPU\n";
}