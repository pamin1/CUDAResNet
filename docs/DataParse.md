# Parsing PyTorch ResNet Model

From the model I am able to extract two files: the npz (model weights) and json (model architecture).
How do I make this readable and send to the GPU?
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

Using nlohmann json, I am able to parse the architecture/layers sizes into the model.
The next step would be to parse the NPZ file similarily to get the layer weights and allocate them to the GPU.
Two options here:
1. allocate during parsing
2. copy weights to host array and then copy to device before launching kernel.

Going with option 2 because its just slightly simpler with the data parsing. It lets me assign to the arrays within the struct without having to allocate memory during the parse. It will take a little more overhead later on though when I do need to copy it over to the GPU.