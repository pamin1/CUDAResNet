#include "CopyModel.h"
#include "ImageClassifier.h"
#include "ModelParse.h"

int main()
{
  // create an image classifier object
  ImageClassifier ic("assets/dog.png");

  // grab the host image
  float *hImage = ic.getHostImage();
  for (int i = 0; i < ic.size; i++) {
    if (hImage[i] == 0) {
      std::cout << "Host image uninitialized?\n"; // shouldnt be any non zero pixels usually
    }
  }
  std::cout << "Host image initialized\n";

  // parse model json
  ModelParse mp("assets/resnet18_manifest.json", "assets/resnet18_fp32.npz");
  ResNet18 model = mp.generateModel();

  mp.printResNet18(model);

  // copy image to GPU
  float *dImage;
  size_t size = 224 * 224 * 3 * sizeof(float);
  cudaMalloc((void **)&dImage, size);
  cudaMemcpy(dImage, hImage, size, cudaMemcpyHostToDevice);

  // copy model to GPU
  CopyModel cm(model);
  ResNetDev dModel = cm.getDevModel();

  while (true)
  {
  }
  return 0;
}
