#include "CopyModel.h"
#include "ImageClassifier.h"
#include "Kernel.cuh"
#include "ModelParse.h"

int main()
{
  // create an image classifier object
  ImageClassifier ic("assets/landscape.png");

  // grab the host image
  float *hImage = ic.getHostImage();
  for (int i = 0; i < ic.size; i++)
  {
    if (hImage[i] == 0)
    {
      std::cout << "Host image uninitialized?\n"; // shouldnt be any non zero pixels usually
    }
  }
  std::cout << "Host image initialized\n";

  // parse model json
  ModelParse mp("assets/resnet18_manifest.json", "assets/resnet18_fp32.npz");
  ResNet18 model = mp.generateModel();

  // mp.printResNet18(model);

  // copy image to GPU
  float *dImage;
  size_t size = 224 * 224 * 3 * sizeof(float);
  CHECK_ERROR(cudaMalloc((void **)&dImage, size));
  cudaMemcpy(dImage, hImage, size, cudaMemcpyHostToDevice);

  // Copy model to GPU
  CopyModel cm(model);
  ResNetDev dModel = cm.getDevModel();

  // initialize output array
  float *out;
  int dim = computeDim(224, 2, 3, dModel.conv1.kernelSize);
  size_t outSize = dim * dim * dModel.conv1.outputSize;
  CHECK_ERROR(cudaMalloc((void **)&out, outSize * sizeof(float)));
  CHECK_ERROR(cudaMemset(out, 0, outSize * sizeof(float)));

  std::cout << dim << "\n";

  // run the first convolution layer
  launchConvKernel(dImage, out, dModel.conv1, dModel.bn1, IMAGE_DIM, 2, 3);
  std::cout << "kernel ran\n";

  // copy the output back
  float *res = (float *)malloc(outSize * sizeof(float));
  CHECK_ERROR(cudaMemcpy(res, out, outSize * sizeof(float), cudaMemcpyDeviceToHost));

  return 0;
}
