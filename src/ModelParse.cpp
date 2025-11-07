#include "ModelParse.h"

ModelParse::ModelParse(std::string path)
{
  std::ifstream f(path);
  this->jsonModel = json::parse(f);
};

ResNet18 ModelParse::generateModel()
{
  ResNet18 model;

  // Initial layers
  model.conv1.outputSize = jsonModel["tensors"]["conv1.weight"][0];
  model.conv1.inputSize = jsonModel["tensors"]["conv1.weight"][1];
  model.conv1.kernelSize = jsonModel["tensors"]["conv1.weight"][2];
  model.bn1.numFeatures = jsonModel["tensors"]["bn1.weight"][0];

  // Helper lambda for parsing blocks
  auto parseBlock = [this](BasicBlock &block, const std::string &prefix)
  {
    block.conv1.outputSize = jsonModel["tensors"][prefix + ".conv1.weight"][0];
    block.conv1.inputSize = jsonModel["tensors"][prefix + ".conv1.weight"][1];
    block.conv1.kernelSize = jsonModel["tensors"][prefix + ".conv1.weight"][2];
    block.bn1.numFeatures = jsonModel["tensors"][prefix + ".bn1.weight"][0];

    block.conv2.outputSize = jsonModel["tensors"][prefix + ".conv2.weight"][0];
    block.conv2.inputSize = jsonModel["tensors"][prefix + ".conv2.weight"][1];
    block.conv2.kernelSize = jsonModel["tensors"][prefix + ".conv2.weight"][2];
    block.bn2.numFeatures = jsonModel["tensors"][prefix + ".bn2.weight"][0];

    // Check if downsample exists
    std::string downsampleKey = prefix + ".downsample.0.weight";
    if (jsonModel["tensors"].contains(downsampleKey))
    {
      block.hasDownsample = true;
      block.downsampleConv.outputSize = jsonModel["tensors"][downsampleKey][0];
      block.downsampleConv.inputSize = jsonModel["tensors"][downsampleKey][1];
      block.downsampleConv.kernelSize = jsonModel["tensors"][downsampleKey][2];
      block.downsampleBn.numFeatures = jsonModel["tensors"][prefix + ".downsample.1.weight"][0];
    }
    else
    {
      block.hasDownsample = false;
    }
  };

  // Parse all layers
  for (int layer = 1; layer <= 4; layer++)
  {
    for (int block = 0; block < 2; block++)
    {
      std::string prefix = "layer" + std::to_string(layer) + "." + std::to_string(block);

      switch (layer)
      {
        case 1:
          parseBlock(model.layer1[block], prefix);
          break;
        case 2:
          parseBlock(model.layer2[block], prefix);
          break;
        case 3:
          parseBlock(model.layer3[block], prefix);
          break;
        case 4:
          parseBlock(model.layer4[block], prefix);
          break;
      }
    }
  }

  // Final FC layer
  model.fc.outputSize = jsonModel["tensors"]["fc.weight"][0];
  model.fc.inputSize = jsonModel["tensors"]["fc.weight"][1];

  return model;
}

void ModelParse::printResNet18(const ResNet18 &model)
{
  std::cout << "=== ResNet18 Architecture ===" << std::endl;
  std::cout << std::endl;

  // Initial layers
  std::cout << "Initial Layers:" << std::endl;
  std::cout << "  conv1: [out=" << model.conv1.outputSize
            << ", in=" << model.conv1.inputSize
            << ", kernel=" << model.conv1.kernelSize << "]" << std::endl;
  std::cout << "  bn1: [features=" << model.bn1.numFeatures << "]" << std::endl;
  std::cout << std::endl;

  // Helper lambda to print a basic block
  auto printBlock = [](const BasicBlock &block, const std::string &name)
  {
    std::cout << "  " << name << ":" << std::endl;
    std::cout << "    conv1: [out=" << block.conv1.outputSize
              << ", in=" << block.conv1.inputSize
              << ", kernel=" << block.conv1.kernelSize << "]" << std::endl;
    std::cout << "    bn1: [features=" << block.bn1.numFeatures << "]" << std::endl;
    std::cout << "    conv2: [out=" << block.conv2.outputSize
              << ", in=" << block.conv2.inputSize
              << ", kernel=" << block.conv2.kernelSize << "]" << std::endl;
    std::cout << "    bn2: [features=" << block.bn2.numFeatures << "]" << std::endl;

    if (block.hasDownsample)
    {
      std::cout << "    downsample:" << std::endl;
      std::cout << "      conv: [out=" << block.downsampleConv.outputSize
                << ", in=" << block.downsampleConv.inputSize
                << ", kernel=" << block.downsampleConv.kernelSize << "]" << std::endl;
      std::cout << "      bn: [features=" << block.downsampleBn.numFeatures << "]" << std::endl;
    }
    else
    {
      std::cout << "    downsample: none" << std::endl;
    }
  };

  // Print all layers
  for (int layer = 1; layer <= 4; layer++)
  {
    std::cout << "Layer " << layer << ":" << std::endl;
    for (int block = 0; block < 2; block++)
    {
      std::string blockName = "block " + std::to_string(block);
      switch (layer)
      {
        case 1:
          printBlock(model.layer1[block], blockName);
          break;
        case 2:
          printBlock(model.layer2[block], blockName);
          break;
        case 3:
          printBlock(model.layer3[block], blockName);
          break;
        case 4:
          printBlock(model.layer4[block], blockName);
          break;
      }
      std::cout << std::endl;
    }
  }

  // Final FC layer
  std::cout << "Final Layers:" << std::endl;
  std::cout << "  fc: [out=" << model.fc.outputSize
            << ", in=" << model.fc.inputSize << "]" << std::endl;
  std::cout << std::endl;

  // Print summary statistics
  std::cout << "=== Summary ===" << std::endl;
  int totalParams = 0;

  // Count parameters
  // Initial conv
  totalParams += model.conv1.outputSize * model.conv1.inputSize *
                 model.conv1.kernelSize * model.conv1.kernelSize;
  totalParams += model.bn1.numFeatures * 4; // weight, bias, mean, var

  // Count for each block
  auto countBlockParams = [&totalParams](const BasicBlock &block)
  {
    totalParams += block.conv1.outputSize * block.conv1.inputSize *
                   block.conv1.kernelSize * block.conv1.kernelSize;
    totalParams += block.bn1.numFeatures * 4;
    totalParams += block.conv2.outputSize * block.conv2.inputSize *
                   block.conv2.kernelSize * block.conv2.kernelSize;
    totalParams += block.bn2.numFeatures * 4;

    if (block.hasDownsample)
    {
      totalParams += block.downsampleConv.outputSize * block.downsampleConv.inputSize *
                     block.downsampleConv.kernelSize * block.downsampleConv.kernelSize;
      totalParams += block.downsampleBn.numFeatures * 4;
    }
  };

  for (int i = 0; i < 2; i++)
  {
    countBlockParams(model.layer1[i]);
    countBlockParams(model.layer2[i]);
    countBlockParams(model.layer3[i]);
    countBlockParams(model.layer4[i]);
  }

  // FC layer
  totalParams += model.fc.outputSize * model.fc.inputSize;
  totalParams += model.fc.outputSize; // bias

  std::cout << "Total parameters: ~" << totalParams / 1000000.0 << "M" << std::endl;
  std::cout << "Memory required (float32): ~" << (totalParams * 4) / (1024.0 * 1024.0) << " MB" << std::endl;
}