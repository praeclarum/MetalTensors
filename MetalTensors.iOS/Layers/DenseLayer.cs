using System;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class DenseLayer : ConvWeightsLayer
    {
        public DenseLayer (int featureChannels, bool bias)
            : base (featureChannels, 1, 1, 1, 1, bias, ConvPadding.Valid)
        {
        }

        public override int[] GetOutputShape (params Tensor[] inputs)
        {
            var inputShape = inputs[0].Shape;
            var outputShape = new int[inputShape.Length];
            Array.Copy (inputShape, outputShape, inputShape.Length);
            outputShape[^1] = FeatureChannels;
            return outputShape;
        }

        protected override MPSNNFilterNode CreateConvWeightsNode (MPSNNImageNode imageNode, MPSCnnConvolutionDataSource convDataSource)
        {
            return new MPSCnnFullyConnectedNode (imageNode, convDataSource);
        }
    }
}
