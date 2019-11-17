using System;
using System.Collections.Concurrent;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public abstract class ConvWeightsLayer : Layer
    {
        public override int MinInputCount => 1;

        public int InFeatureChannels => Weights.InChannels;
        public int OutFeatureChannels => Weights.OutChannels;
        public int SizeX => Weights.SizeX;
        public int SizeY => Weights.SizeY;
        public int StrideX => Weights.StrideX;
        public int StrideY => Weights.StrideY;
        public bool Bias => Weights.Bias;
        public WeightsInit WeightsInit => Weights.WeightsInit;
        public float BiasInit => Weights.BiasInit;
        public ConvPadding Padding { get; }

        public ConvWeights Weights { get; }

        protected ConvWeightsLayer (int inFeatureChannels, int outFeatureChannels, int sizeX, int sizeY, int strideX, int strideY, ConvPadding padding, bool bias, WeightsInit weightsInit, float biasInit)
        {
            Weights = new ConvWeights (Label, inFeatureChannels, outFeatureChannels, sizeX, sizeY, strideX, strideY, bias, weightsInit, biasInit);
            Padding = padding;
        }

        protected override MPSNNFilterNode CreateFilterNode ((MPSNNImageNode ImageNode, int[] Shape)[] inputs, IMTLDevice device)
        {
            return CreateConvWeightsNode (inputs[0].ImageNode, GetMetalConvDataSource (device));
        }

        public override MPSCnnConvolutionDataSource GetMetalConvDataSource (IMTLDevice device)
        {
            return Weights.GetDataSource (device);
        }

        protected abstract MPSNNFilterNode CreateConvWeightsNode (MPSNNImageNode imageNode, MPSCnnConvolutionDataSource convDataSource);

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
