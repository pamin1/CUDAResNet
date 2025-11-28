#include "ImageParse.h"
#include "Kernel.cuh"
#include "ModelImplementation.h"
#include "ModelParse.h"
#include <chrono>
#include <fmt/format.h>

constexpr int sampleSize = 1000;
constexpr int imgSize = 224 * 224 * 3;
constexpr size_t byteSize = imgSize * sizeof(float);

int main()
{
    // parse model
    ModelParse mp("assets/resnet18_manifest.json", "assets/resnet18_fp32.npz");
    ResNet18 model = mp.generateModel();
    mp.printResNet18(model);

    // load images onto cpu
    std::vector<float> h_images(imgSize * sampleSize);
    for (int i = 0; i < sampleSize; i++)
    {
        std::string path = fmt::format("assets/cifar10/images/image_{:04d}.png", i);
        ImageParse im = ImageParse(path);

        float *h_image = im.getHostImage();

        std::memcpy(&h_images[i * imgSize], h_image, byteSize);
    }
    std::cout << "Loaded images\n";

    // load classes into map
    std::unordered_map<int, std::string> map;
    std::ifstream file("assets/imagenet_classes.txt");
    std::string line;
    int i = 0;

    while (std::getline(file, line))
    {
        map[i] = line;
        i++;
    }

    file.close();

    // copy over to gpu
    float *d_images;
    CHECK_ERROR(cudaMalloc((void **)&d_images, byteSize * sampleSize));
    cudaMemcpy(d_images, h_images.data(), byteSize * sampleSize, cudaMemcpyHostToDevice);

    // gpu warmup
    for (int i = 0; i < 50; i++)
    {
        float *conf = launchModel(model, d_images + (i * imgSize));
    }

    // run benchmarking
    double totalTime = 0;
    for (int i = 0; i < sampleSize; i++)
    {
        const auto start = std::chrono::high_resolution_clock::now();
        float *conf = launchModel(model, d_images + (i * imgSize));
        int predicted_class = 0;
        float max_score = conf[0];
        for (int i = 1; i < 1000; i++)
        {
            if (conf[i] > max_score)
            {
                max_score = conf[i];
                predicted_class = i;
            }
        }
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> duration = end - start;
        totalTime += duration.count();

        // uncomment for inference verification
        std::string img = fmt::format("image_{:04d}", i);
        // std::cout << fmt::format("{:<12} Class: {:<30} Confidence: {:>8.4f}  Time: {:>7.2f} ms\n", img, map[predicted_class], max_score, duration.count());
    }

    // total timing metrics
    std::cout << "Total Time: " << totalTime << " ms\n"
              << "Average Time: " << totalTime / sampleSize << " ms/img\n"
              << "Average Freq: " << 1000 * sampleSize / totalTime << " Hz\n";

    // cleanup
    mp.freeModel(model);
    cudaFree(d_images);
    return 0;
}