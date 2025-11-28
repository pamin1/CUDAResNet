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