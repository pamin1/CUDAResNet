#include <fstream>
#include <iostream>
#include <vector>

class CIFARLoader
{
private:
  std::vector<unsigned char> labels;
  std::vector<std::vector<unsigned char>> images; // Raw 32×32×3 data

public:
  CIFARLoader(const std::string &filepath, int maxImages = -1)
  {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open())
    {
      std::cerr << "Failed to open CIFAR file: " << filepath << std::endl;
      return;
    }

    int count = 0;
    while (file.good() && (maxImages < 0 || count < maxImages))
    {
      // Read label (1 byte)
      unsigned char label;
      file.read((char *)&label, 1);
      if (!file.good())
        break;

      // Read image (3072 bytes)
      std::vector<unsigned char> img(3072);
      file.read((char *)img.data(), 3072);
      if (!file.good())
        break;

      labels.push_back(label);
      images.push_back(img);
      count++;
    }

    file.close();
    std::cout << "Loaded " << images.size() << " CIFAR-10 images\n";
  }

  // Convert CIFAR image (32×32×3, CHW format) to ResNet format (224×224×3, HWC format)
  float *getProcessedImage(int idx)
  {
    if (idx >= images.size())
      return nullptr;

    const auto &img = images[idx];
    float *processed = new float[224 * 224 * 3];

    // CIFAR format: 1024 red values, then 1024 green, then 1024 blue (32×32 each)
    // We need to: 1) Resize 32×32 to 224×224, 2) Convert to HWC, 3) Normalize

    for (int h = 0; h < 224; h++)
    {
      for (int w = 0; w < 224; w++)
      {
        // Simple nearest-neighbor upsampling
        int src_h = h * 32 / 224;
        int src_w = w * 32 / 224;
        int src_idx = src_h * 32 + src_w;

        // Convert CHW to HWC and normalize to [0, 1]
        processed[h * 224 * 3 + w * 3 + 0] = img[src_idx] / 255.0f;        // R
        processed[h * 224 * 3 + w * 3 + 1] = img[1024 + src_idx] / 255.0f; // G
        processed[h * 224 * 3 + w * 3 + 2] = img[2048 + src_idx] / 255.0f; // B
      }
    }

    return processed;
  }

  unsigned char getLabel(int idx)
  {
    return labels[idx];
  }

  int size()
  {
    return images.size();
  }
};