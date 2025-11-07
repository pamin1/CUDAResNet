#include <cnpy/cnpy.h>
#include <fstream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

struct ConvLayer
{
  int outputSize;
  int inputSize;
  int kernelSize;
  float *d_weight;
};

struct BatchNorm
{
  int numFeatures;
  float *d_weight;
  float *d_bias;
  float *d_runningMean;
  float *d_runningVar;
};

struct FullyConnected
{
  int outputSize;
  int inputSize;
  float *d_weight;
  float *d_bias;
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

  ResNet18 generateModel();
  void printResNet18(const ResNet18 &model);

private:
  json jsonModel;
  cnpy::npz_t npzData;
  ResNet18 model;
};
