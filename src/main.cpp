#include "ImageClassifier.h"
#include "ModelParse.h"

int main()
{
  // create an image classifier object
  // ImageClassifier ic("assets/dog.png");

  // grab the host image
  // std::vector<float> hostImage = ic.getHostImage();
  // if (hostImage.size() != 0) {
  //     std::cout << "successful host image grab\n";
  // }

  // parse model json
  ModelParse mp("assets/resnet18_manifest.json", "assets/resnet18_fp32.npz");
  json m = mp.getModel();

  ResNet18 model = mp.generateModel();
  mp.printResNet18(model);

  return 0;
}
