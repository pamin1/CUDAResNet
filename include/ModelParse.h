#ifndef MODEL_PARSE_H
#define MODEL_PARSE_H

#include <cnpy/cnpy.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <ResNetDev.h>
using json = nlohmann::json;

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

#endif