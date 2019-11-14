using System;
using System.Collections.Concurrent;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public abstract class ConvWeightsLayer : Layer
    {
        public override int InputCount => 1;

        public int FeatureChannels { get; }
        public int SizeX { get; }
        public int SizeY { get; }
        public int StrideX { get; }
        public int StrideY { get; }
        public bool Bias { get; }
        public ConvPadding Padding { get; }

        public ConvWeightsLayer (int featureChannels, int sizeX, int sizeY, int strideX, int strideY, bool bias, ConvPadding padding)
        {
            if (featureChannels <= 0)
                throw new ArgumentOutOfRangeException (nameof (featureChannels), "Number of feature channels must be > 0");

            FeatureChannels = featureChannels;
            SizeX = sizeX;
            SizeY = sizeY;
            StrideX = strideX;
            StrideY = strideY;
            Bias = bias;
            Padding = padding;
        }

        protected override MPSNNFilterNode CreateFilterNode ((MPSNNImageNode ImageNode, int[] Shape)[] inputs, IMTLDevice device)
        {
            var input = inputs[0];
            int inChannels = input.Shape[^1];
            return CreateConvWeightsNode (input.ImageNode, GetWeights (inChannels, device));
        }

        public override MPSCnnConvolutionDataSource? GetMetalConvDataSource (IMTLDevice device)
        {
            var key = device.Handle;
            if (deviceWeights.TryGetValue (key, out var w))
                return w;
            return null;
        }

        protected abstract MPSNNFilterNode CreateConvWeightsNode (MPSNNImageNode imageNode, MPSCnnConvolutionDataSource convDataSource);

        readonly ConcurrentDictionary<IntPtr, ConvWeights> deviceWeights =
            new ConcurrentDictionary<IntPtr, ConvWeights> ();

        ConvWeights GetWeights (int inChannels, IMTLDevice device)
        {
            var key = device.Handle;
            if (deviceWeights.TryGetValue (key, out var w))
                return w;

            w = new ConvWeights (inChannels, FeatureChannels, SizeX, SizeY, StrideX, StrideY, Bias, Label, device);

            if (deviceWeights.TryAdd (key, w))
                return w;
            return deviceWeights[key];
        }

        public static int ConvOutputLength (int inputLength, int size, int stride, ConvPadding padding, int dilation)
        {
            if (inputLength < 0)
                throw new ArgumentOutOfRangeException (nameof (inputLength), "Convolution input dimension must be >= 0");

            var dilatedFilterSize = (size - 1) * dilation + 1;
            var outputLength = padding switch
            {
                ConvPadding.Same => inputLength,
                _ => inputLength - dilatedFilterSize + 1
            };
            var r = (outputLength + stride - 1) / stride;
            return r;
        }
    }
}
