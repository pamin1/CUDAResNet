#ifndef RESNETDEV_H
#define RESNETDEV_H

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

#endif