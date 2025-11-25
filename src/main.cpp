#include "ModelParse.h"
#include "ImageClassifier.h"
#include "Kernel.cuh"
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

    mp.printResNet18(model);

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "GPU Memory - Free: " << free_mem / (1024.0 * 1024.0)
              << " MB, Total: " << total_mem / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Used: " << (total_mem - free_mem) / (1024.0 * 1024.0) << " MB" << std::endl;

    // // copy image to GPU
    float *dImage;
    size_t size = 224 * 224 * 3 * sizeof(float);
    CHECK_ERROR(cudaMalloc((void **)&dImage, size));
    cudaMemcpy(dImage, hImage, size, cudaMemcpyHostToDevice);

    // // initialize output array
    float *out;
    int dim = computeDim(IMAGE_DIM, 2, 3, model.conv1.kernelSize);
    size_t outSize = dim * dim * model.conv1.outputSize;
    CHECK_ERROR(cudaMalloc((void **)&out, outSize * sizeof(float)));
    CHECK_ERROR(cudaMemset(out, 0, outSize * sizeof(float)));

    // MODEL IMPLEMENT
    // TODO: push this all into a function(s)

    // ------------------------------------------
    // conv1 + bn1
    // conv1: 224×224×3 to 112×112×64
    launchConvKernel(dImage, out, model.conv1, model.bn1, IMAGE_DIM, 2, 3);
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
    std::cout << "Conv1 + MaxPool complete\n";

    // ------------------------------------------
    // LAYER 1
    // block 0
    runBasicBlock(model.layer1[0], out, out,
                  64, // inputChannels
                  56, // inputH
                  56, // inputW
                  1); // stride=1
    cudaDeviceSynchronize();
    std::cout << "Layer1.0 complete\n";

    // block 1
    runBasicBlock(model.layer1[1], out, out,
                  64, // inputChannels
                  56, // inputH
                  56, // inputW
                  1); // stride=1
    cudaDeviceSynchronize();
    std::cout << "Layer1.1 complete\n";

    // ------------------------------------------
    // LAYER 2
    // block 0 (with downsample)
    runBasicBlock(model.layer2[0], out, out,
                  64, // inputChannels
                  56, // inputH
                  56, // inputW
                  2); // stride=2 (downsample)
    cudaDeviceSynchronize();
    std::cout << "Layer2.0 complete\n";

    // block 1
    runBasicBlock(model.layer2[1], out, out,
                  128, // inputChannels
                  28,  // inputH
                  28,  // inputW
                  1);  // stride=1
    cudaDeviceSynchronize();
    std::cout << "Layer2.1 complete\n";

    // ------------------------------------------
    // LAYER 3
    // block 0 (with downsample)
    runBasicBlock(model.layer3[0], out, out,
                  128, // inputChannels
                  28,  // inputH
                  28,  // inputW
                  2);  // stride=2 (downsample)
    cudaDeviceSynchronize();
    std::cout << "Layer3.0 complete\n";

    // block 1
    runBasicBlock(model.layer3[1], out, out,
                  256, // inputChannels
                  14,  // inputH
                  14,  // inputW
                  1);  // stride=1
    cudaDeviceSynchronize();
    std::cout << "Layer3.1 complete\n";

    // ------------------------------------------
    // LAYER 4
    // block 0 (with downsample)
    runBasicBlock(model.layer4[0], out, out,
                  256, // inputChannels
                  14,  // inputH
                  14,  // inputW
                  2);  // stride=2 (downsample)
    cudaDeviceSynchronize();
    std::cout << "Layer4.0 complete\n";

    // block 1
    runBasicBlock(model.layer4[1], out, out,
                  512, // inputChannels
                  7,   // inputH
                  7,   // inputW
                  1);  // stride=1
    cudaDeviceSynchronize();
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

    launchFCKernel(pooled_out, final_out, model.fc, 512, 1000);
    cudaDeviceSynchronize();
    std::cout << "FC complete\n";

    // ------------------------------------------
    // copy back
    float *h_results = new float[1000];
    cudaMemcpy(h_results, final_out, 1000 * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<std::string> labels = loadImageNetLabels("assets/imagenet_classes.txt");

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