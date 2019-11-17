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
            Padding = ConvPadding.Same;
        }

        public override void ValidateInputShapes (params Tensor[] inputs)
        {
            base.ValidateInputShapes (inputs);

            var inputShape = inputs[0].Shape;
            if (inputShape.Length < 3)
                throw new ArgumentException ($"Pooling inputs must have 3 dimensions HxWxC ({inputs.Length} given)", nameof (inputs));
        }

        public override int[] GetOutputShape (params Tensor[] inputs)
        {
            // https://github.com/keras-team/keras/blob/afff7b4326f380a54c73400d1e2ae03890162bdf/keras/layers/pooling.py#L180

            var inputShape = inputs[0].Shape;
            var h = inputShape[0];
            var w = inputShape[1];
            var fc = inputShape[^1];
            var kh = ConvWeightsLayer.ConvOutputLength (h, SizeY, StrideY, ConvPadding.Same, 1);
            var kw = ConvWeightsLayer.ConvOutputLength (w, SizeX, StrideX, ConvPadding.Same, 1);
            //var sh = kh / StrideY;
            //var sw = kw / StrideX;
            return new[] { kh, kw, fc };
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
