using System;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class MaxPoolLayer : PoolLayer
    {
        public MaxPoolLayer (int sizeX, int sizeY, int strideX, int strideY, ConvPadding padding, string? name = null)
            : base (sizeX, sizeY, strideX, strideY, padding, name: name)
        {
        }

        public MaxPoolLayer (int size = 2, int stride = 2, ConvPadding padding = ConvPadding.Valid)
            : this (size, size, stride, stride, padding)
        {
        }

        protected override MPSNNFilterNode CreatePoolNode (MPSNNImageNode imageNode)
        {
            return new MPSCnnPoolingMaxNode (imageNode, (nuint)SizeX, (nuint)SizeY, (nuint)StrideX, (nuint)StrideY);
        }
    }
}
