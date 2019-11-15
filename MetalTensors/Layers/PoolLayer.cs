using System;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public abstract class PoolLayer : Layer
    {
        public override int InputCount => 1;

        public int SizeX { get; }
        public int SizeY { get; }
        public int StrideX { get; }
        public int StrideY { get; }

        protected PoolLayer (int sizeX, int sizeY, int strideX, int strideY)
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
        }

        protected override MPSNNFilterNode CreateFilterNode ((MPSNNImageNode ImageNode, int[] Shape)[] inputs, IMTLDevice device)
        {
            return CreatePoolNode (inputs[0].ImageNode);
        }

        protected abstract MPSNNFilterNode CreatePoolNode (MPSNNImageNode imageNode);
    }
}