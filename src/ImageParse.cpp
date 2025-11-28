#include "ImageParse.h"

ImageParse::ImageParse(std::string path)
{
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);

    // Check if the image was loaded successfully
    if (image.empty())
    {
        std::cerr << "Error: Could not open or find the image." << std::endl;
        return;
    }

    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);

    // match PT resizing
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(256, 256), 0, 0, cv::INTER_LINEAR);

    // center cropping
    int cropX = (256 - 224) / 2; // = 16
    int cropY = (256 - 224) / 2; // = 16
    cv::Rect cropRegion(cropX, cropY, 224, 224);
    cv::Mat cropped = resized(cropRegion);

    // scale
    cropped.convertTo(cropped, CV_32FC3, 1.0 / 255.0);

    // normalize
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float stdv[3] = {0.229f, 0.224f, 0.225f};

    for (int h = 0; h < 224; ++h)
    {
        auto row = cropped.ptr<cv::Vec3f>(h);
        for (int w = 0; w < 224; ++w)
        {
            size_t base = h * 224 + w;
            host[0 * 224 * 224 + base] = (row[w][0] - mean[0]) / stdv[0]; // R
            host[1 * 224 * 224 + base] = (row[w][1] - mean[1]) / stdv[1]; // G
            host[2 * 224 * 224 + base] = (row[w][2] - mean[2]) / stdv[2]; // B
        }
    }
}