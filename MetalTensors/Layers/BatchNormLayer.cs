using System;
using System.Collections.Concurrent;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class BatchNormLayer : Layer
    {
        public override int MinInputCount => 1;

        public int FeatureChannels { get; }
        public BatchNormWeights Weights { get; }

        public BatchNormLayer (int featureChannels)
        {
            Weights = new BatchNormWeights (Label, featureChannels);
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
