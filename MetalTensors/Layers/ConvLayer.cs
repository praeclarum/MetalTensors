using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class ConvLayer : ConvWeightsLayer
    {
        public ConvLayer (int featureChannels, int sizeX, int sizeY, int strideX, int strideY, bool bias, float biasInit, ConvPadding padding)
            : base (featureChannels, sizeX, sizeY, strideX, strideY, bias, biasInit, padding)
        {
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
            return new MPSCnnConvolutionNode (imageNode, convDataSource);
        }
    }
}
