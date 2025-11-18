#include "CopyModel.h"
#include "ImageClassifier.h"
#include "Kernel.cuh"
#include "ModelParse.h"

void printWeightStatistics(const float *d_weight, int out_channels, int in_channels,
                           int kernel_height, int kernel_width)
{
  int total_weights = out_channels * in_channels * kernel_height * kernel_width;

  float *h_weight = (float *)malloc(total_weights * sizeof(float));
  cudaMemcpy(h_weight, d_weight, total_weights * sizeof(float), cudaMemcpyDeviceToHost);

  printf("\n=== Weight Statistics ===\n");
  printf("Total weights: %d\n", total_weights);

  int weights_per_channel = in_channels * kernel_height * kernel_width;

  for (int oc = 0; oc < out_channels; oc++)
  {
    float min_val = h_weight[oc * weights_per_channel];
    float max_val = h_weight[oc * weights_per_channel];
    float sum = 0.0f;
    int zero_count = 0;

    for (int i = 0; i < weights_per_channel; i++)
    {
      int idx = oc * weights_per_channel + i;
      float val = h_weight[idx];

      if (val < min_val)
        min_val = val;
      if (val > max_val)
        max_val = val;
      if (val == 0.0f)
        zero_count++;
      sum += val;
    }

    printf("Ch %2d: min=%8.4f, max=%8.4f, mean=%8.4f, zeros=%d/%d (%.1f%%)\n", oc, min_val, max_val,
           sum / weights_per_channel, zero_count, weights_per_channel,
           100.0f * zero_count / weights_per_channel);
  }

  free(h_weight);
}

void dumpWeightsRaw(const float *d_weight, int total_weights, const char *filename)
{
  float *h_weight = (float *)malloc(total_weights * sizeof(float));
  cudaMemcpy(h_weight, d_weight, total_weights * sizeof(float), cudaMemcpyDeviceToHost);

  FILE *fp = fopen(filename, "w");
  for (int i = 0; i < total_weights; i++)
  {
    fprintf(fp, "%.8f\n", h_weight[i]);
  }
  fclose(fp);
  free(h_weight);

  printf("Dumped %d weights to %s\n", total_weights, filename);
}

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

  // try to see the data?
  int count = 0;
  int i = 0;
  // std::ofstream outfile("output.txt", std::ios::app);
  // for (int i = 0; i < outSize; i++)
  // {
  //   outfile << res[i] << "\n";
  //   count++;
  // }
  // outfile.close();
  std::cout << "Count: " << count << "\n";

  printf("Channel 4 weights [0:10]: ");
  for (int i = 0; i < 10; i++)
  {
    printf("%f ", model.conv1.h_weight[4 * dModel.conv1.inputSize * dModel.conv1.kernelSize *
                                           dModel.conv1.kernelSize +
                                       i]);
  }
  printf("\n");

  dumpWeightsRaw(dModel.conv1.d_weight, 64 * 3 * 7 * 7, "weights.txt");

  return 0;
}
