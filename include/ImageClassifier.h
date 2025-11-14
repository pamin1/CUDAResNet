#include <iostream>
#include <opencv2/opencv.hpp>

class ImageClassifier
{
public:
  explicit ImageClassifier(std::string path); // only allow constrcutor with
                                              // image path (no default)
  std::vector<float> getHostImage()
  {
    return host;
  }

private:
  std::vector<float> host;
};