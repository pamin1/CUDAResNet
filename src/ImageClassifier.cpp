#include "ImageClassifier.h"

ImageClassifier::ImageClassifier(std::string path) {
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);

    // Check if the image was loaded successfully
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image." << std::endl;
        return;
    }
    std::cout << "successful image load\n";

    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);

    // resize+crop to (224,224) as needed...
    // convert to float [0,1]
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

    cv::imshow("rgb", rgb);
    cv::waitKey(0);

    // normalize + HWC->NCHW
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float stdv[3] = {0.229f, 0.224f, 0.225f};

    this->host.resize(3 * 224 * 224);
    for (int h = 0; h < 224; ++h) {
        auto row = rgb.ptr<cv::Vec3f>(h);
        for (int w = 0; w < 224; ++w) {
            size_t base = h * 224 + w;
            this->host[0 * 224 * 224 + base] = (row[w][0] - mean[0]) / stdv[0]; // R
            this->host[1 * 224 * 224 + base] = (row[w][1] - mean[1]) / stdv[1]; // G
            this->host[2 * 224 * 224 + base] = (row[w][2] - mean[2]) / stdv[2]; // B
        }
    }
};