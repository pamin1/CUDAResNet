# Parsing PyTorch ResNet Model

From the model we are able to extract two files: the npz (model weights) and json (model architecture).
How do we make this readable and send to the GPU?
* There are convolution layers, batch norms, downsamples, and a fully connected layer
* 5 total layers
  * Layer 0: convolution, batch norm
 
  * Layer 1.0: 2x (convolution, batch norm)
  * Layer 1.1: 2x (convolution, batch norm)
 
  * Layer 2.0: (convolution, batch norm), downsample, (convolution, batch norm)
  * Layer 2.1: 2x (convolution, batch norm)
 
  * Layer 3.0: (convolution, batch norm), downsample, (convolution, batch norm)
  * Layer 3.1: 2x (convolution, batch norm)

  * Layer 4.0: (convolution, batch norm), downsample, (convolution, batch norm)
  * Layer 4.1: 2x (convolution, batch norm)

Ends with Fully Connected layer.

What is the structure of each component:
Convolution Layer:
* weight: [Filters, Channels, Height, Width]
  * Filters: number of output channels
  * Channels: number of input channels
  * Width x Height: size of the convolution kernel

Batch Norm Layer:
* weight
* bias
* running_mean
* running_var
* num_batches_tracked

Downsample Layer:
* weight: [Filters, Channels, 1, 1] (1x1 Convolution)
* bias
* running_mean
* running_var
* num_batches_tracked

Fully Connected Layer:
* weight: [Filters, Channels]
* bias