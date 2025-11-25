#ifndef MODEL_PARSE_H
#define MODEL_PARSE_H

#include <ResNetDev.h>
#include <cnpy/cnpy.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

class ModelParse
{
  public:
    ModelParse(std::string jsonPath, std::string npzPath);
    ResNet18 generateModel();
    void freeModel(ResNet18 &model);

    void printResNet18(const ResNet18 &model);

    json getModel()
    {
        return jsonModel;
    }

    cnpy::npz_t getData()
    {
        return npzData;
    }

  private:
    json jsonModel;
    cnpy::npz_t npzData; // prevents dangling pointers later on
    ResNet18 model;
};

#endif