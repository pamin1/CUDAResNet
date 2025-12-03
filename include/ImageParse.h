/**
 * Author: Prachit Amin
 * ECE 4122
 * 12/1/2025
 * Declares ImageParse class for preprocessing image.
 */

#include <iostream>
#include <opencv2/opencv.hpp>

class ImageParse
{
  public:
    /**
     * @brief Image parsing object that preprocesses and stores the input image
     * @param path File path to the chosen input image
     */
    explicit ImageParse(std::string path);

    /**
     * @brief Getter for the input image float array
     * @return Array of floats (224x224*3)
     */
    float *getHostImage()
    {
        return host;
    }
    static const size_t size = 224 * 224 * 3;

  private:
    float host[size];
};