#include "ModelParse.h"
#include "util.h"
#include "iostream"

struct ConvLayerDev
{
  int outputSize;
  int inputSize;
  int kernelSize; // guaranteed square kernel shapes
  const float *d_weight;
};

struct BatchNormDev
{
  int numFeatures;
  const float *d_weight;
  const float *d_bias;
  const float *d_runningMean;
  const float *d_runningVar;
};

struct FullyConnectedDev
{
  int outputSize;
  int inputSize;
  const float *d_weight;
  const float *d_bias;
};

struct DownsampleDev
{
  ConvLayerDev weight; // should be a 1x1 conv layer
  BatchNormDev bn;
};

struct BasicBlockDev
{
  ConvLayerDev conv1;
  BatchNormDev bn1;
  ConvLayerDev conv2;
  BatchNormDev bn2;

  bool hasDownsample;
  DownsampleDev ds;
};

struct ResNetDev
{
  // Initial layer
  ConvLayerDev conv1;
  BatchNormDev bn1;

  // 4 stages, each with 2 BasicBlocks
  BasicBlockDev layer1[2];
  BasicBlockDev layer2[2];
  BasicBlockDev layer3[2];
  BasicBlockDev layer4[2];

  // Final classifier
  FullyConnectedDev fc;
};

class CopyModel
{
public:
  CopyModel(const ResNet18 &model);

  void copyConvLayer(ConvLayerDev &dst, const ConvLayer &src);
  void copyBatchNorm(BatchNormDev &dst, const BatchNorm &src);
  void copyDownSample(DownsampleDev &dst, const Downsample &src);
  void copyFullyConnected(FullyConnectedDev &dst, const FullyConnected &src);
  void copyBasicBlock(BasicBlockDev &dst, const BasicBlock &src);

  ResNetDev getDevModel()
  {
    return devModel;
  }

private:
  ResNetDev devModel;
};