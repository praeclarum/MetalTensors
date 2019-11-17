using System;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class ConvLayer : ConvWeightsLayer
    {
        static readonly MPSNNDefaultPadding samePadding = MPSNNDefaultPadding.Create (
            MPSNNPaddingMethod.AddRemainderToTopLeft | MPSNNPaddingMethod.AlignCentered | MPSNNPaddingMethod.SizeSame);
        static readonly MPSNNDefaultPadding validPadding = MPSNNDefaultPadding.Create (
            MPSNNPaddingMethod.AddRemainderToTopLeft | MPSNNPaddingMethod.AlignCentered | MPSNNPaddingMethod.SizeValidOnly);

        public ConvLayer (int featureChannels, int sizeX, int sizeY, int strideX, int strideY, ConvPadding padding, bool bias, WeightsInit weightsInit, float biasInit)
            : base (featureChannels, sizeX, sizeY, strideX, strideY, padding, bias, weightsInit, biasInit)
        {
        }

        public override void ValidateInputShapes (params Tensor[] inputs)
        {
            base.ValidateInputShapes (inputs);

            var inputShape = inputs[0].Shape;
            if (inputShape.Length < 3)
                throw new ArgumentException ($"Conv inputs must have 3 dimensions HxWxC ({inputs.Length} given)", nameof (inputs));
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

        protected override MPSNNFilterNode CreateConvWeightsNode (MPSNNImageNode imageNode, MPSCnnConvolutionDataSource convDataSource)
        {
            return new MPSCnnConvolutionNode (imageNode, convDataSource) {
                PaddingPolicy = Padding == ConvPadding.Same ? samePadding : validPadding,
            };
        }
    }
}
