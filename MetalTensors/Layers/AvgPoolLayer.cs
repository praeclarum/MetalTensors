using System;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class AvgPoolLayer : PoolLayer
    {
        public AvgPoolLayer (int sizeX, int sizeY, int strideX, int strideY, ConvPadding padding, string? name = null)
            : base (sizeX, sizeY, strideX, strideY, padding, name)
        {
        }

        public AvgPoolLayer (int size = 2, int stride = 2, ConvPadding padding = ConvPadding.Valid)
            : this (size, size, stride, stride, padding)
        {
        }

        protected override MPSNNFilterNode CreatePoolNode (MPSNNImageNode imageNode)
        {
            return new MPSCnnPoolingAverageNode (imageNode, (nuint)SizeX, (nuint)SizeY, (nuint)StrideX, (nuint)StrideY);
        }
    }
}
