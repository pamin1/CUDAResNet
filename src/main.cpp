#include "CopyModel.h"
#include "ImageClassifier.h"
#include "Kernel.cuh"
#include "ModelParse.h"
#include <labels.h>

int main()
{
  // create an image classifier object
  ImageClassifier ic("assets/dog.png");

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
  CopyModel cm;
  ResNetDev dModel = cm.getDevModel();

  // initialize output array
  float *out;
  int dim = computeDim(IMAGE_DIM, 2, 3, model.conv1.kernelSize);
  size_t outSize = dim * dim * model.conv1.outputSize;
  CHECK_ERROR(cudaMalloc((void **)&out, outSize * sizeof(float)));
  CHECK_ERROR(cudaMemset(out, 0, outSize * sizeof(float)));

  // MODEL IMPLEMENT
  // TODO: push this all into a function(s)

  // ------------------------------------------
  // conv1 + bn1
  cm.copyConvLayer(dModel.conv1, model.conv1);
  cm.copyBatchNorm(dModel.bn1, model.bn1);

  // conv1: 224×224×3 to 112×112×64
  launchConvKernel(dImage, out, dModel.conv1, dModel.bn1, IMAGE_DIM, 2, 3);
  cudaDeviceSynchronize();

  // ReLU
  launchReLUKernel(out, out, 64 * 112 * 112);
  cudaDeviceSynchronize();

  // max pool: 112×112×64 to 56×56×64
  float *temp_pool;
  cudaMalloc(&temp_pool, 64 * 56 * 56 * sizeof(float));
  launchMaxPoolKernel(out, temp_pool, 112, 112, 64, 3, 2, 1);
  cudaDeviceSynchronize();

  // copy pooled output back to main buffer
  cudaMemcpy(out, temp_pool, 64 * 56 * 56 * sizeof(float), cudaMemcpyDeviceToDevice);

  // free this layer
  cudaFree(temp_pool);
  cm.freeConvLayer(dModel.conv1);
  cm.freeBatchNorm(dModel.bn1);
  std::cout << "Conv1 + MaxPool complete\n";

  // ------------------------------------------
  // LAYER 1
  // block 0
  cm.copyBasicBlock(dModel.layer1[0], model.layer1[0]);
  runBasicBlock(dModel.layer1[0], out, out,
                64, // inputChannels
                56, // inputH
                56, // inputW
                1); // stride=1
  cudaDeviceSynchronize();
  cm.freeBasicBlock(dModel.layer1[0]);
  std::cout << "Layer1.0 complete\n";

  // block 1
  cm.copyBasicBlock(dModel.layer1[1], model.layer1[1]);
  runBasicBlock(dModel.layer1[1], out, out,
                64, // inputChannels
                56, // inputH
                56, // inputW
                1); // stride=1
  cudaDeviceSynchronize();
  cm.freeBasicBlock(dModel.layer1[1]);
  std::cout << "Layer1.1 complete\n";

  // ------------------------------------------
  // LAYER 2
  // block 0 (with downsample)
  cm.copyBasicBlock(dModel.layer2[0], model.layer2[0]);
  runBasicBlock(dModel.layer2[0], out, out,
                64, // inputChannels
                56, // inputH
                56, // inputW
                2); // stride=2 (downsample)
  cudaDeviceSynchronize();
  cm.freeBasicBlock(dModel.layer2[0]);
  std::cout << "Layer2.0 complete\n";

  // block 1
  cm.copyBasicBlock(dModel.layer2[1], model.layer2[1]);
  runBasicBlock(dModel.layer2[1], out, out,
                128, // inputChannels
                28,  // inputH
                28,  // inputW
                1);  // stride=1
  cudaDeviceSynchronize();
  cm.freeBasicBlock(dModel.layer2[1]);
  std::cout << "Layer2.1 complete\n";

  // ------------------------------------------
  // LAYER 3
  // block 0 (with downsample)
  cm.copyBasicBlock(dModel.layer3[0], model.layer3[0]);
  runBasicBlock(dModel.layer3[0], out, out,
                128, // inputChannels
                28,  // inputH
                28,  // inputW
                2);  // stride=2 (downsample)
  cudaDeviceSynchronize();
  cm.freeBasicBlock(dModel.layer3[0]);
  std::cout << "Layer3.0 complete\n";

  // block 1
  cm.copyBasicBlock(dModel.layer3[1], model.layer3[1]);
  runBasicBlock(dModel.layer3[1], out, out,
                256, // inputChannels
                14,  // inputH
                14,  // inputW
                1);  // stride=1
  cudaDeviceSynchronize();
  cm.freeBasicBlock(dModel.layer3[1]);
  std::cout << "Layer3.1 complete\n";

  // ------------------------------------------
  // LAYER 4
  // block 0 (with downsample)
  cm.copyBasicBlock(dModel.layer4[0], model.layer4[0]);
  runBasicBlock(dModel.layer4[0], out, out,
                256, // inputChannels
                14,  // inputH
                14,  // inputW
                2);  // stride=2 (downsample)
  cudaDeviceSynchronize();
  cm.freeBasicBlock(dModel.layer4[0]);
  std::cout << "Layer4.0 complete\n";

  // block 1
  cm.copyBasicBlock(dModel.layer4[1], model.layer4[1]);
  runBasicBlock(dModel.layer4[1], out, out,
                512, // inputChannels
                7,   // inputH
                7,   // inputW
                1);  // stride=1
  cudaDeviceSynchronize();
  cm.freeBasicBlock(dModel.layer4[1]);
  std::cout << "Layer4.1 complete\n";

  // ADAPTIVE AVERAGE POOL
  float *pooled_out;
  cudaMalloc(&pooled_out, 512 * sizeof(float));

  launchAdaptiveAvgPoolKernel(out, pooled_out, 7, 7, 512);
  cudaDeviceSynchronize();
  std::cout << "AdaptiveAvgPool complete\n";

  // ------------------------------------------
  // FULLY CONNECTED
  float *final_out;
  cudaMalloc(&final_out, 1000 * sizeof(float));

  cm.copyFullyConnected(dModel.fc, model.fc);
  launchFCKernel(pooled_out, final_out, dModel.fc, 512, 1000);
  cudaDeviceSynchronize();
  cm.freeFullyConnected(dModel.fc);
  std::cout << "FC complete\n";

  // ------------------------------------------
  // copy back
  float *h_results = new float[1000];
  cudaMemcpy(h_results, final_out, 1000 * sizeof(float), cudaMemcpyDeviceToHost);

  std::vector<std::string> labels = loadImageNetLabels("imagenet_classes.txt");

  // find top-5 predictions
  std::cout << "\nTop-5 Predictions:\n";
  for (int i = 0; i < 5; i++)
  {
    float max_val = -INFINITY;
    int max_idx = -1;
    for (int j = 0; j < 1000; j++)
    {
      if (h_results[j] > max_val)
      {
        max_val = h_results[j];
        max_idx = j;
      }
    }
    std::cout << max_idx << ": " << labels[max_idx] << " (score: " << max_val << ")\n";
    h_results[max_idx] = -INFINITY;
  }

  // Cleanup
  delete[] h_results;
  cudaFree(pooled_out);
  cudaFree(final_out);
  // ------------------------------------------

  return 0;
}