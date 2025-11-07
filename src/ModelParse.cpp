#include "ModelParse.h"

ModelParse::ModelParse(std::string path)
{
  std::ifstream f(path);
  this->model = json::parse(f);
};