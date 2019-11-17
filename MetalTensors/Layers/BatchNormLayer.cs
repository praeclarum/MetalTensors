using System;
using System.Collections.Concurrent;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class BatchNormLayer : Layer
    {
        public override int MinInputCount => 1;

        public override int[] GetOutputShape (params Tensor[] inputs)
        {
            return inputs[0].Shape;
        }

        protected override MPSNNFilterNode CreateFilterNode ((MPSNNImageNode ImageNode, int[] Shape)[] inputs, IMTLDevice device)
        {
            var inputShape = inputs[0].Shape;
            return new MPSCnnBatchNormalizationNode (inputs[0].ImageNode, GetWeights (inputShape[^1], device));
        }

        readonly ConcurrentDictionary<IntPtr, BatchNormWeights> deviceWeights =
            new ConcurrentDictionary<IntPtr, BatchNormWeights> ();

        BatchNormWeights GetWeights (int inChannels, IMTLDevice device)
        {
            var key = device.Handle;
            if (deviceWeights.TryGetValue (key, out var w))
                return w;

            w = new BatchNormWeights (inChannels, Label, device);

            if (deviceWeights.TryAdd (key, w))
                return w;
            return deviceWeights[key];
        }
    }
}
