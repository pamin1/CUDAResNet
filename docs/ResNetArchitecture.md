# ResNet Architecture

Purpose of ResNet? As researchers added more layers to other models, the performance of the models decreased. This is the vanishing gradient problem.

Vanishing Gradient Problem:
* During training, the model identifies errors and adjusts the weights using back-propagation.
* As we move backwards through the model, we compute the gradients of the lost w.r.t to each weight and update them accordingly (Gradient Descent Algorithm)
* As more layers are attached to the model, the result of gradient descent through back-propagation becomes negligible. 
* Since the early layers are foundational to overall feature detection, if those layers are poorly trained, the model will have worse performance due to early errors.

How does ResNet address this? By adding Residual/Skip connections (layer bypassing). Generally, the $(n-1)^{th}$ layer bypasses the $n^{th}$ layer, and adds to the output of the $(n+1)^{th}$ . Why is this important? There are two cases: the $n^{th}$ layer provides information or it doesn't. In either case, we use the input to the $n^{th}$, because it should provide useful information. This way whether the $n^{th}$ layer is useful or not, we should get some more information from the previous layer, allowing in consistent or improved performance, even as the number of layers increases. 

TL,DR: ResNet will maintain or improve model performance by allowing more layers to provide information deeper in the network, improving upon typical CNN performance.