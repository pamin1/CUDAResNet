#include "Kernel.cuh"
#include "ResNetDev.h"
#include <iostream>

float *launchModel(const ResNet18 &model, const float *image, float *out)
{
    launchConvKernel((float *)image, out, model.conv1, model.bn1, IMAGE_DIM, 2, 3);
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

    // copy back
    float *h_results = new float[1000];
    cudaMemcpy(h_results, final_out, 1000 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(pooled_out);
    cudaFree(final_out);

    return h_results;
};