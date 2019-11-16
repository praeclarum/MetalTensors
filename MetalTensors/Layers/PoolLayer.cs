using System;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public abstract class PoolLayer : Layer
    {
        static readonly MPSNNDefaultPadding samePadding = MPSNNDefaultPadding.CreatePaddingForTensorflowAveragePooling ();
        static readonly MPSNNDefaultPadding validPadding = MPSNNDefaultPadding.CreatePaddingForTensorflowAveragePoolingValidOnly ();

        public override int InputCount => 1;

        public int SizeX { get; }
        public int SizeY { get; }
        public int StrideX { get; }
        public int StrideY { get; }
        public ConvPadding Padding { get; }

        protected PoolLayer (int sizeX, int sizeY, int strideX, int strideY, ConvPadding padding = ConvPadding.Same)
        {
            if (sizeX < 1)
                throw new ArgumentOutOfRangeException (nameof (sizeX), "Pooling width must be > 0");
            if (sizeY < 1)
                throw new ArgumentOutOfRangeException (nameof (sizeY), "Pooling height must be > 0");
            if (strideX < 1)
                throw new ArgumentOutOfRangeException (nameof (sizeX), "Pooling x stride must be > 0");
            if (strideY < 1)
                throw new ArgumentOutOfRangeException (nameof (sizeY), "Pooling y stride must be > 0");

            SizeX = sizeX;
            SizeY = sizeY;
            StrideX = strideX;
            StrideY = strideY;
            Padding = padding;
        }

        protected override MPSNNFilterNode CreateFilterNode ((MPSNNImageNode ImageNode, int[] Shape)[] inputs, IMTLDevice device)
        {
            var node = CreatePoolNode (inputs[0].ImageNode);
            node.PaddingPolicy = Padding == ConvPadding.Same ? samePadding : validPadding;
            return node;
        }

        protected abstract MPSNNFilterNode CreatePoolNode (MPSNNImageNode imageNode);
    }
}
