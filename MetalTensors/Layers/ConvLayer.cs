﻿using System;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class ConvLayer : ConvWeightsLayer
    {
        static readonly MPSNNDefaultPadding samePadding = MPSNNDefaultPadding.Create (
            MPSNNPaddingMethod.AddRemainderToTopLeft | MPSNNPaddingMethod.AlignCentered | MPSNNPaddingMethod.SizeSame);
        static readonly MPSNNDefaultPadding validPadding = MPSNNDefaultPadding.Create (
            MPSNNPaddingMethod.AddRemainderToTopLeft | MPSNNPaddingMethod.AlignCentered | MPSNNPaddingMethod.SizeValidOnly);

        public ConvLayer (int inFeatureChannels, int outFeatureChannels, int sizeX, int sizeY, int strideX, int strideY, ConvPadding padding, bool bias, WeightsInit weightsInit, float biasInit, string? name = null, bool isTrainable = true)
            : base (inFeatureChannels, outFeatureChannels, sizeX, sizeY, strideX, strideY, padding, bias, weightsInit, biasInit, name, isTrainable: isTrainable)
        {
        }

        public override void ValidateInputShapes (params Tensor[] inputs)
        {
            base.ValidateInputShapes (inputs);

            foreach (var i in inputs) {
                var inputShape = i.Shape;
                if (inputShape.Length != 3)
                    throw new ArgumentException ($"Conv inputs must have 3 dimensions HxWxC ({inputs.Length} given)", nameof (inputs));
                if (inputShape[^1] != InFeatureChannels)
                    throw new ArgumentException ($"Expected conv input with {InFeatureChannels} channels, but got {inputShape[^1]}", nameof (inputs));
            }
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
            return new[] { kh, kw, OutFeatureChannels };
        }

        protected override MPSNNFilterNode CreateConvWeightsNode (MPSNNImageNode imageNode, MPSCnnConvolutionDataSource convDataSource)
        {
            return new MPSCnnConvolutionNode (imageNode, convDataSource) {
                PaddingPolicy = Padding == ConvPadding.Same ? samePadding : validPadding,
            };
        }
    }
}
