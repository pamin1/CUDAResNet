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

class ModelParse
{
public:
  ModelParse(std::string path);
  json getModel()
  {
    return model;
  }

private:
  json model;
};