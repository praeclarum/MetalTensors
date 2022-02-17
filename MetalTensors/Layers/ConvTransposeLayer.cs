using System;
using Foundation;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    /// <summary>
    /// When the stride in any dimension is greater than 1, the convolution transpose puts (stride - 1) zeroes in-between the source 
    /// image pixels to create an expanded image.Then a convolution is done over the expanded image to generate the output of the
    /// convolution transpose.
    /// Intermediate image size = (srcSize - 1) * Stride + 1
    /// </summary>
    public class ConvTransposeLayer : ConvWeightsLayer
    {
        // Most MPS neural network filters are considered forward filters.
        // Some (for example, convolution transpose and unpooling) are considered reverse filters.
        // For the reverse filters, the image stride is measured in destination values rather than source values
        // and has the effect of enlarging the image rather than reducing it.
        // When a reverse filter is used to "undo" the effects of a forward filter,
        // the size policy should be the opposite of the forward padding method.
        // For example, if the forward filter used MPSNNPaddingMethodSizeValidOnly | MPSNNPaddingMethodAddRemainderToTopLeft,
        // the reverse filter should use MPSNNPaddingMethodSizeFull | MPSNNPaddingMethodAddRemainderToTopLeft.
        // Some consideration of the geometry of inputs and outputs will reveal why this is so.
        // It is usually not important to adjust the centering method
        // because the size of the reverse result generally doesn't suffer from centering asymmetries.

        static readonly IMPSNNPadding samePadding1 = new SamePadding (1);
        static readonly IMPSNNPadding samePadding2 = new SamePadding (2);
        static readonly IMPSNNPadding samePadding3 = new SamePadding (3);
        static readonly IMPSNNPadding samePadding4 = new SamePadding (4);
        static readonly IMPSNNPadding validPadding = MPSNNDefaultPadding.Create (
            MPSNNPaddingMethod.AddRemainderToTopLeft | MPSNNPaddingMethod.AlignCentered | MPSNNPaddingMethod.SizeFull);

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
            var padding = validPadding;
            if (Padding == ConvPadding.Same) {
                padding = StrideX switch {
                    1 => samePadding1,
                    2 => samePadding2,
                    3 => samePadding3,
                    4 => samePadding4,
                    var x => throw new InvalidOperationException ($"Cannot create convolution transpose with stride {x}"),
                };
            }
            return new MPSCnnConvolutionTransposeNode (imageNode, gradientStateNode, convDataSource) {
                PaddingPolicy = padding,
            };
        }

        class SamePadding : NSObject, IMPSNNPadding
        {
            public MPSNNPaddingMethod PaddingMethod => MPSNNPaddingMethod.AddRemainderToTopLeft | MPSNNPaddingMethod.AlignCentered | MPSNNPaddingMethod.Custom;

            public nuint Stride { get; }

            public SamePadding (int stride)
            {
                Stride = (nuint)stride;
            }

            public void EncodeTo (NSCoder encoder)
            {
            }

            [Export ("destinationImageDescriptorForSourceImages:sourceStates:forKernel:suggestedDescriptor:")]
            public MPSImageDescriptor GetImageDescriptor (MPSImage[] images, MPSState[] states, MPSKernel kernel, MPSImageDescriptor desc)
            {
                desc.Width = images[0].Width * Stride;
                desc.Height = images[0].Height * Stride;
                return desc;
            }
        }
    }
}
