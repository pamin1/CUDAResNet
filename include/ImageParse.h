#include <iostream>
#include <opencv2/opencv.hpp>

class ImageParse
{
public:
  explicit ImageParse(std::string path); // only allow constrcutor with
                                              // image path (no default)
  float *getHostImage()
  {
    return host;
  }
  static const size_t size = 224 * 224 * 3;

private:
  float host[size];
};