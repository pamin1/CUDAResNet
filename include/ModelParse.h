/**
 * Author: Prachit Amin
 * ECE 4122
 * 12/1/2025
 * Declares ModelParse class.
 */

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
    /**
     * @brief Loads the model architecture and weights into standard formats
     * @param jsonPath Path to ResNet manifest in json form
     * @param npzPath Path to ResNet weights in npz form
     */
    ModelParse(std::string jsonPath, std::string npzPath);

    /**
     * @brief Populates structs with model weights. Allocates and copies the weights to GPU as well
     */
    ResNet18 generateModel();

    /**
     * @brief Free all memory allocated on GPU by the ResNet model
     * @param model Model to be freed
     */
    void freeModel(ResNet18 &model);

    /**
     * @brief Prints out the model architecture to understand the model and verify the data has loaded correctly
     * @param model Struct of information storing parsed input files
     */
    void printResNet18(const ResNet18 &model);

    /**
     * @brief Getter for jsonModel
     * @return Json type object
     */
    json getModel()
    {
        return jsonModel;
    }

    /**
     * @brief Getter for npz data structure
     * @return Dictionary of npyArray objects
     */
    cnpy::npz_t getData()
    {
        return npzData;
    }

  private:
    json jsonModel;
    cnpy::npz_t npzData;
    ResNet18 model;
};

#endif