using System;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class MaxPoolLayer : PoolLayer
    {
        public MaxPoolLayer (int sizeX, int sizeY, int strideX, int strideY)
            : base (sizeX, sizeY, strideX, strideY)
        {
        }

        public MaxPoolLayer (int size = 2, int stride = 2)
            : this (size, size, stride, stride)
        {
        }

        public override int[] GetOutputShape (params Tensor[] inputs)
        {
            // https://github.com/keras-team/keras/blob/afff7b4326f380a54c73400d1e2ae03890162bdf/keras/layers/pooling.py#L180

            var inputShape = inputs[0].Shape;
            var h = inputShape[0];
            var w = inputShape[1];
            var fc = inputShape[^1];
            var kh = ConvLayer.ConvOutputLength (h, SizeY, StrideY, ConvPadding.Valid, 1);
            var kw = ConvLayer.ConvOutputLength (w, SizeX, StrideX, ConvPadding.Valid, 1);
            //var sh = kh / StrideY;
            //var sw = kw / StrideX;
            return new[] { kh, kw, fc };
        }

        protected override MPSNNFilterNode CreatePoolNode (MPSNNImageNode imageNode)
        {
            return new MPSCnnPoolingMaxNode (imageNode, (nuint)SizeX, (nuint)SizeY, (nuint)StrideX, (nuint)StrideY);
        }
    }
}
