using System;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class DenseLayer : Layer
    {
        public override int InputCount => 1;

        public int FeatureChannels { get; }

        public DenseLayer (int featureChannels)
        {
            FeatureChannels = featureChannels;
        }

        public override int[] GetOutputShape (params Tensor[] inputs)
        {
            var inputShape = inputs[0].Shape;
            var outputShape = new int[inputShape.Length];
            Array.Copy (inputShape, outputShape, inputShape.Length);
            outputShape[^1] = FeatureChannels;
            return outputShape;
        }

        protected override MPSNNFilterNode CreateFilterNode ((MPSNNImageNode ImageNode, int[] Shape)[] inputs, IMTLDevice device)
        {
            var input = inputs[0];
            int inChannels = input.Shape[^1];
            return new MPSCnnFullyConnectedNode (input.ImageNode, GetWeights (inChannels, device));
        }

        ConvWeights GetWeights (int inChannels, IMTLDevice device)
        {
            var w = new ConvWeights (inChannels, FeatureChannels, 1, 1, 1, 1, true, Label, device);
            return w;
        }
    }
}
