using System;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class DenseLayer : ConvWeightsLayer
    {
        public DenseLayer (int featureChannels, int sizeX, int sizeY, bool bias, float biasInit)
            : base (featureChannels, sizeX, sizeY, 1, 1, bias, biasInit, ConvPadding.Valid)
        {
        }

        public override int[] GetOutputShape (params Tensor[] inputs)
        {
            var inputShape = inputs[0].Shape;
            var outputShape = new int[inputShape.Length];
            for (var i = 0; i < inputShape.Length; i++) {
                var s = inputShape[i];
                if (i == 0) {
                    s /= SizeY;
                }
                else if (i == 1) {
                    s /= SizeX;
                }
                outputShape[i] = s;
            }
            outputShape[^1] = FeatureChannels;
            return outputShape;
        }

        protected override MPSNNFilterNode CreateConvWeightsNode (MPSNNImageNode imageNode, MPSCnnConvolutionDataSource convDataSource)
        {
            return new MPSCnnFullyConnectedNode (imageNode, convDataSource);
        }
    }
}
