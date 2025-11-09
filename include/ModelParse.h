#include <cnpy/cnpy.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct ConvLayer
{
  int outputSize;
  int inputSize;
  int kernelSize; // guaranteed square kernel shapes
  const float *h_weight;
};

struct BatchNorm
{
  int numFeatures;
  const float *h_weight;
  const float *h_bias;
  const float *h_runningMean;
  const float *h_runningVar;
};

struct FullyConnected
{
  int outputSize;
  int inputSize;
  const float *h_weight;
  const float *h_bias;
};

struct Downsample
{
  ConvLayer weight; // should be a 1x1 conv layer
  BatchNorm bn;
};

struct BasicBlock
{
  ConvLayer conv1;
  BatchNorm bn1;
  ConvLayer conv2;
  BatchNorm bn2;

  bool hasDownsample;
  ConvLayer downsampleConv;
  BatchNorm downsampleBn;
};

struct ResNet18
{
  // Initial layer
  ConvLayer conv1;
  BatchNorm bn1;

  // 4 stages, each with 2 BasicBlocks
  BasicBlock layer1[2];
  BasicBlock layer2[2];
  BasicBlock layer3[2];
  BasicBlock layer4[2];

  // Final classifier
  FullyConnected fc;
};

class ModelParse
{
public:
  ModelParse(std::string jsonPath, std::string npzPath);

  json getModel()
  {
    return jsonModel;
  }

  cnpy::npz_t getData()
  {
    return npzData;
  }

  ResNet18 generateModel();
  void printResNet18(const ResNet18 &model);

private:
  json jsonModel;
  cnpy::npz_t npzData; // prevents dangling pointers later on
  ResNet18 model;
};
