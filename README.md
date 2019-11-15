# MetalTensors

MetalTensors provides a high-level API over Apple's Metal Performance Shaders
neural network library. It allows you to train and execute neural networks
on iPhones, iPads, and Macs.



## Quick intro

The main abstraction is the `Tensor` object.
These are hyperdimensional matrices of numbers.
You can index into them to get values and
you can combine them to perform computations.

```csharp
using MetalTensors;

var a = Tensor.Constant(40.0f);
var b = Tensor.Constant(2.0f);
var answer = a + b;

Console.WriteLine(answer[0]);
```

Prints:

```
42
```



## Architecture

There are four main classes offered with increasing levels of abstraction:

* `Tensors` represent primitive bags of data
* `Layers` represent operations on tensors
* `Models` combine layers into groups to make complex computations
* `Applications` use models to perform a particular task

### Multi-device support

Tensors and layers and models do their best to stay
metal device agnostic.

The goal is to make it possible to train using multiple GPUs
on the same machine.

Note, while training the same model
on multiple GPUs is technically allowed by this library,
it is not recommended since weight updates will be unstable.

Instead, multiple GPUs can be used to train different models.
This is useful for hyperparameter tuning.

To use the non-default GPU, pass an `IMTLDevice` to the
`Train` method of `Model`.



## Why another nerual network library?

Most neural network libraries aren't able to use the GPUs
on iPhones, iPads, and Macs.
Also, most libraries are for Python and not .NET.

I wanted to be able to easily play with new network architectures
safe in the knowledge that the work I do will run perfectly
on mobile devices.

