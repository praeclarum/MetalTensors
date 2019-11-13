using System;
using Metal;
using MetalPerformanceShaders;

using static MetalTensors.MetalExtensions;

namespace MetalTensors.Layers
{
    public class ConvLayer : Layer
    {
        public override int InputCount => 1;

        public int FeatureChannels { get; }
        public int SizeX { get; }
        public int SizeY { get; }
        public int StrideX { get; }
        public int StrideY { get; }
        public ConvPadding Padding { get; }

        public ConvLayer (int featureChannels, int sizeX, int sizeY, int strideX, int strideY, ConvPadding padding)
        {
            if (featureChannels <= 0)
                throw new ArgumentOutOfRangeException (nameof (featureChannels), "Convolution 2D feature channels must be > 0");

            FeatureChannels = featureChannels;
            SizeX = sizeX;
            SizeY = sizeY;
            StrideX = strideX;
            StrideY = strideY;
            Padding = padding;
        }

        public ConvLayer (int featureChannels, int size, int stride = 1, ConvPadding padding = ConvPadding.Same)
            : this (featureChannels, size, size, stride, stride, padding)
        {
        }

        public override int[] GetOutputShape (params Tensor[] inputs)
        {
            // https://github.com/keras-team/keras/blob/f06524c44e5f6926968cb2bb3ddd1e523f5474c5/keras/utils/conv_utils.py#L85

            var inputShape = inputs[0].Shape;
            var h = inputShape[0];
            var w = inputShape[1];
            var kh = ConvOutputLength (h, SizeY, StrideY, Padding, 1);
            var kw = ConvOutputLength (w, SizeX, StrideX, Padding, 1);
            //var sh = kh / StrideY;
            //var sw = kw / StrideX;
            return new[] { kh, kw, FeatureChannels };
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

        protected override MPSNNFilterNode CreateFilterNode ((MPSNNImageNode ImageNode, int[] Shape)[] inputs, IMTLDevice device)
        {
            var input = inputs[0];
            int inChannels = input.Shape[^1];
            return new MPSCnnConvolutionNode (input.ImageNode, GetWeights (inChannels, device));
        }

        ConvWeights GetWeights (int inChannels, IMTLDevice device)
        {
            var w = new ConvWeights (inChannels, FeatureChannels, SizeX, SizeY, StrideX, StrideY, true, Label, device);
            return w;
        }
    }
}
