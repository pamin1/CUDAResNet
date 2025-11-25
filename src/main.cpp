#include "Benchmark.h"
#include "ImageClassifier.h"
#include "Kernel.cuh"
#include "ModelImplementation.h"
#include "ModelParse.h"
#include <labels.h>

#define SAMPLE 1000

int main()
{
    // parse model
    ModelParse mp("assets/resnet18_manifest.json", "assets/resnet18_fp32.npz");
    ResNet18 model = mp.generateModel();

    mp.printResNet18(model);

    // initialize CIFAR
    CIFARLoader cifar("assets/cifar-10-batches-bin/test_batch.bin", SAMPLE);

    // SETUP: Preload all images to GPU (do this ONCE, before benchmarking)
    std::cout << "Loading images to GPU...\n";
    std::vector<float *> d_images;
    std::vector<float *> h_results_buffers;

    for (int i = 0; i < SAMPLE; i++)
    {
        float *h_image = cifar.getProcessedImage(i);

        float *d_image;
        CHECK_ERROR(cudaMalloc((void **)&d_image, 224 * 224 * 3 * sizeof(float)));
        cudaMemcpy(d_image, h_image, 224 * 224 * 3 * sizeof(float), cudaMemcpyHostToDevice);

        d_images.push_back(d_image);
        h_results_buffers.push_back(new float[1000]);

        delete[] h_image;
    }

    std::cout << "All images loaded to GPU\n";

    // WARMUP
    std::cout << "Warming up...\n";
    for (int i = 0; i < 10; i++)
    {
        float *res = launchModel(model, d_images[i % d_images.size()]);
        delete[] res;
    }

    // BENCHMARK (pure inference, no I/O)
    std::cout << "Running benchmark...\n";
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < SAMPLE; i++)
    {
        float *res = launchModel(model, d_images[i]);
        delete[] res;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms;
    cudaEventElapsedTime(&total_ms, start, stop);

    std::cout << "\n=== BENCHMARK RESULTS (Pure Inference) ===\n";
    std::cout << "Images: SAMPLE\n";
    std::cout << "Total time: " << total_ms << " ms\n";
    std::cout << "Average per image: " << total_ms / SAMPLE << " ms\n";
    std::cout << "Throughput: " << (SAMPLE * 1000.0f) / total_ms << " img/s\n";

    // CLEANUP
    for (auto d_img : d_images)
        cudaFree(d_img);
    for (auto h_res : h_results_buffers)
        delete[] h_res;

    // cleanup
    mp.freeModel(model);

    return 0;
}