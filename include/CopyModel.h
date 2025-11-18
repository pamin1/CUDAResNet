#include "ModelParse.h"
#include "ResNetDev.h"
#include "iostream"
#include "util.h"

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