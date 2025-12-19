# Kernel Implementation

## Objective
Run ResNet layers on the input image in GPU.

The image stays on the GPU; need to have the modified data across layers.
Due to local memory size of GPU, I need to move layers on and off of GPU.

So the ideal control flow would be:
1. Copy Image
2. Copy Layer n
3. Convolve
4. Save Image (in place?)
5. Free Layer n
6. Repeat from Step 2

Have to be careful of operations on the image. Pixels are stored as a 1D row-major array.

Each thread will handle a single pixel in the output image. So each thread needs to handle 
the convolution of the pixels around it 

## Optimizations
The most obvious optimization is to use shared memory for the convolution kernel. I need to consider that there is c_in input channels and c_out output channels for each conv layer. Each output pixel needs k_h * k_w * c_in input values, c_out number of feature maps compute simultaneously.

Loading all channels into shared memory at once would be too large. Tiling the channels themselves would make the most sense. So I would need the GPU to do a chunk of each channel. The channels are contiguous in memory so I would need to account for that as well.
So each feature map will be H x W, each thread block will provide tile_h x tile_w outputs.
tile dims are determined with consideration for padding.

Each block will perform one tile of the convolution. Further, threadIdx.x and threadIdx.y create thread cooperation within the thread. Global position tells where in the image we are. 