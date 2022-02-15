using System;
using System.Collections.Concurrent;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class BatchNormLayer : Layer
    {
        // https://github.com/apple/turicreate/blob/98b61f551f26429e05025c79785d7b4b0ef50295/src/ml/neural_net/mps_layers.mm#L443
        public const float DefaultEpsilon = 0.001f;

        public override int MinInputCount => 1;

        public int FeatureChannels { get; }
        public BatchNormWeights Weights { get; }

        public BatchNormLayer (int featureChannels, float epsilon = DefaultEpsilon)
        {
            Weights = new BatchNormWeights (Name, featureChannels, epsilon);
            FeatureChannels = featureChannels;
        }

        public override int[] GetOutputShape (params Tensor[] inputs)
        {
            return inputs[0].Shape;
        }

        protected override MPSNNFilterNode CreateFilterNode ((MPSNNImageNode ImageNode, int[] Shape)[] inputs, IMTLDevice device)
        {
            return new MPSCnnBatchNormalizationNode (inputs[0].ImageNode, Weights.GetDataSource (device));
        }
    }
}
