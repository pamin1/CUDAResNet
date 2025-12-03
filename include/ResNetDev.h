/**
 * Author: Prachit Amin
 * ECE 4122
 * 12/1/2025
 * Declares ResNet model and sublayers/sections
 */

#ifndef RESNETDEV_H
#define RESNETDEV_H

struct ConvLayer
{
    int outputSize;
    int inputSize;
    int kernelSize; // guaranteed square kernel shapes
    const float *d_weight;
};

struct BatchNorm
{
    int numFeatures;
    const float *d_weight;
    const float *d_bias;
    const float *d_runningMean;
    const float *d_runningVar;
};

struct FullyConnected
{
    int outputSize;
    int inputSize;
    const float *d_weight;
    const float *d_bias;
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
    Downsample ds;
};

struct ResNet18
{
    // initial layer
    ConvLayer conv1;
    BatchNorm bn1;

    // 4 stages, each with 2 BasicBlocks
    BasicBlock layer1[2];
    BasicBlock layer2[2];
    BasicBlock layer3[2];
    BasicBlock layer4[2];

    // final classifier
    FullyConnected fc;
};

#endif