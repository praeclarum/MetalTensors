using System;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class ConvTransposeLayer : ConvWeightsLayer
    {
        static readonly MPSNNDefaultPadding samePadding = MPSNNDefaultPadding.Create (
            MPSNNPaddingMethod.AddRemainderToTopLeft | MPSNNPaddingMethod.AlignCentered | MPSNNPaddingMethod.SizeSame);
        static readonly MPSNNDefaultPadding validPadding = MPSNNDefaultPadding.Create (
            MPSNNPaddingMethod.AddRemainderToTopLeft | MPSNNPaddingMethod.AlignCentered | MPSNNPaddingMethod.SizeValidOnly);

        public ConvTransposeLayer (int inFeaureChannels, int outFeatureChannels, int sizeX, int sizeY, int strideX, int strideY, ConvPadding padding, bool bias, WeightsInit weightsInit, float biasInit)
            : base (inFeaureChannels, outFeatureChannels, sizeX, sizeY, strideX, strideY, padding, bias, weightsInit, biasInit)
        {
        }

        public override void ValidateInputShapes (params Tensor[] inputs)
        {
            base.ValidateInputShapes (inputs);

            foreach (var i in inputs) {
                var inputShape = i.Shape;
                if (inputShape.Length != 3)
                    throw new ArgumentException ($"Conv transpose inputs must have 3 dimensions HxWxC ({inputs.Length} given)", nameof (inputs));
                if (inputShape[^1] != InFeatureChannels)
                    throw new ArgumentException ($"Expected conv transpose input with {InFeatureChannels} channels, but got {inputShape[^1]}", nameof (inputs));
            }
        }

        public override int[] GetOutputShape (params Tensor[] inputs)
        {
            var inputShape = inputs[0].Shape;
            var h = inputShape[0];
            var w = inputShape[1];
            var kh = ConvTransposeOutputLength (h, SizeY, StrideY, Padding, 1, null);
            var kw = ConvTransposeOutputLength (w, SizeX, StrideX, Padding, 1, null);
            return new[] { kh, kw, OutFeatureChannels };
        }

        protected override MPSNNFilterNode CreateConvWeightsNode (MPSNNImageNode imageNode, MPSCnnConvolutionDataSource convDataSource)
        {
            MPSCnnConvolutionGradientStateNode? gradientStateNode = null;
            return new MPSCnnConvolutionTransposeNode (imageNode, gradientStateNode, convDataSource) {
                PaddingPolicy = Padding == ConvPadding.Same ? samePadding : validPadding,
            };
        }
    }
}
