#include "ImageClassifier.h"

int main() {
    // create an image classifier object
    ImageClassifier ic("images/dog.png");

    // grab the host image
    std::vector<float> hostImage = ic.getHostImage();
    if (hostImage.size() != 0) {
        std::cout << "successful host image grab\n";
    }
    
    // begin CUDA allocation
    
    return 0;
}