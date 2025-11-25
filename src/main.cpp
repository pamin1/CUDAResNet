#include "ImageClassifier.h"
#include "Kernel.cuh"
#include "ModelImplementation.h"
#include "ModelParse.h"
#include <labels.h>

int main()
{
    // create an image classifier object
    ImageClassifier ic("assets/dog.png");

    // grab the host image
    float *input = ic.getHostImage();
    for (int i = 0; i < ic.size; i++)
    {
        if (input[i] == 0)
        {
            std::cout << "Host image uninitialized?\n"; // shouldnt be any non zero pixels usually
        }
    }
    std::cout << "Host image initialized\n";

    // parse model json
    ModelParse mp("assets/resnet18_manifest.json", "assets/resnet18_fp32.npz");
    ResNet18 model = mp.generateModel();

    mp.printResNet18(model);

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "\nGPU Memory - Free: " << free_mem / (1024.0 * 1024.0)
              << " MB, Total: " << total_mem / (1024.0 * 1024.0) << " MB"
              << "\n";
    std::cout << "Used: " << (total_mem - free_mem) / (1024.0 * 1024.0) << " MB"
              << "\n";

    // copy image to GPU
    float *image;
    size_t size = 224 * 224 * 3 * sizeof(float);

    CHECK_ERROR(cudaMalloc((void **)&image, size));
    cudaMemcpy(image, input, size, cudaMemcpyHostToDevice);

    // initialize output array
    float *out;
    int dim = computeDim(IMAGE_DIM, 2, 3, model.conv1.kernelSize);
    size_t outSize = dim * dim * model.conv1.outputSize;

    CHECK_ERROR(cudaMalloc((void **)&out, outSize * sizeof(float)));
    CHECK_ERROR(cudaMemset(out, 0, outSize * sizeof(float)));

    // launch the CUDA ResNet18 model
    float *res = launchModel(model, image, out);

    // find top-5 predictions
    std::cout << "\nTop-5 Predictions:\n";

    std::vector<std::string> labels = loadImageNetLabels("assets/imagenet_classes.txt");
    for (int i = 0; i < 5; i++)
    {
        float max_val = -INFINITY;
        int max_idx = -1;
        for (int j = 0; j < 1000; j++)
        {
            if (res[j] > max_val)
            {
                max_val = res[j];
                max_idx = j;
            }
        }
        std::cout << max_idx << ": " << labels[max_idx] << " (score: " << max_val << ")\n";
        res[max_idx] = -INFINITY;
    }

    // cleanup
    delete[] res;
    mp.freeModel(model);

    return 0;
}