# CUDA ResNet

## Objective
Explore and implement the ResNet architecture using CUDA acceleration. 

## Usage
Clone:
```
git clone https://github.com/pamin1/CUDAResNet.git
cd CUDAResNet
git submodule update --init --recursive
```

Build:
```
mkdir build && cd build
cmake ..
make
```

Run:
```
./resnet
```

## Performance Benchmarks
### Accuracy
Using the same weights across all test groups so the inferences are deterministic, thus accuracy testing will be redundant.

| Implementation                    | Latency (ms/img) | Throughput (img/s) | vs PyTorch CUDA |
| :-------------------------------- | :--------------: | :----------------: | :-------------: |
| **Baseline: PyTorch CUDA**        |      4.281       |       233.58       |    Baseline     |
| **Baseline: PyTorch CPU**         |      18.550      |       53.92        |  4.33x slower   |
| Custom CUDA - Shared Memory Image |      34.744      |       28.78        |  8.11x slower   |
| Custom CUDA - Naive               |      39.208      |       25.51        |  9.16x slower   |

**v1.1**: Shared Memory Image stores only the image in shared memory. There is mild reuse of the image data so it makes sense there is not as extreme of a speed up. The real speed likely remains to be seen through storing the kernel weights.
Better optimizations will be seen by improving the data reuse pattern.

**v1.0**: Naive implementation consistently ~9x slower than CUDA accelerated PyTorch, only 2x slower than PyTorch CPU. Removed variable sample size; implemented warmup of 50 inferences before beginning timing and commited to 1000 samples for testing.
## Why ResNet?
### Vanishing Gradient Problem
As the number of layers in a model increase, the performance of the models decreases:
* During training, the model identifies errors and adjusts the weights using back-propagation.
* As the model move backwards through the model, the model compute the gradients of the lost w.r.t to each weight and update them accordingly (Gradient Descent Algorithm)
* As more layers are attached to the model, the result of gradient descent through back-propagation becomes negligible. 
* Since the early layers are foundational to overall feature detection, if those layers are poorly trained, the model will have worse performance due to early errors.

### How does ResNet address this? 
By adding Residual/Skip connections (layer bypassing). Generally, the $(n-1)^{th}$ layer bypasses the $n^{th}$ layer, and adds to the output of the $(n+1)^{th}$ . 

*Why is this important?* There are two cases: the $n^{th}$ layer provides information or it doesn't. In either case, the model can use the input to the $n^{th}$, because it should provide useful information. This way whether the $n^{th}$ layer is useful or not, the model should get some more information from the previous layer, allowing in consistent/improved performance, as the number of layers increases. 

TL,DR: ResNet will maintain or improve model performance by allowing more layers to provide information deeper in the network, improving upon typical CNN performance.

## Implementation Organization
The implementation handles the entire inference stack, from image preprocessing to top-k results.

### Image Preprocessing
The first layer of the ResNet18 architecture uses a 224x224x3 convolution. What this means in context is that the input data is a 224x224 array with 3 channels. In the context of an image this would mean a 224x224 input image, and split into R, G, and B channels. 

I used OpenCV to parse the input image. It converts an arbitrarily sized input image into the required 224x224 size and scales the uchar[0-255] to floats[0-1], matching the required input format for the model weights.

### Model Parsing
The model is split between two files. The JSON manifest contains the model architecture describing the shapes and sizes of each layer. The npz file contains the actual weights used for kernel operations at each layer. Each layer is copied into a host structure that contains the layers sizes as well as a float array of the weights.

5 total layers:

0. Layer 0: convolution, batch norm
1. Layer 1:
   1. 2x (convolution, batch norm)
   2. 2x (convolution, batch norm)
2. Layer 2:
   1. 2x (convolution, batch norm) + DS
   2. 2x (convolution, batch norm)
3. Layer 3:
   1. 2x (convolution, batch norm) + DS
   2. 2x (convolution, batch norm)
4. Layer 4:
   1. 2x (convolution, batch norm) + DS
   2. 2x (convolution, batch norm)
5. Fully Connected

Using nlohmann json, I am able to parse the architecture/layers sizes into the model.
I parse the NPZ file similarily to get the layer weights and allocate them to the GPU.

On the first version, I had two separate structs for holding the model data. One struct held the data on the host and the other on the device. This added a ton of bloat to the codebase and implementation.

After refactoring, the memory allocation step has been integrated into one setup function such that the actual implementation does not require copying and freeing later on. This all fits on the comfortably on the GPU since the ResNet18 architecture is relatively small (~11M params, ~44MB). Similar to initialization, freeing is also all done in one function.

### Model Implementation
The implementation runs the layers in the logical order defined by the architecture. I use hardcoded values for striding and padding.
