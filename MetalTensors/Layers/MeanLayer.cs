using System;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class MeanLayer : ReductionLayer
    {
        public override int[] GetOutputShape (params Tensor[] inputs)
        {
            var shape = inputs[0].Shape;
            var n = shape.Length;
            var r = new int[n];
            for (var i = 0; i < n - 1; i++) {
                r[i] = shape[i];
            }
            r[^1] = 1;
            return r;
        }

        protected override MPSNNFilterNode CreateReductionNode (MPSNNImageNode imageNode)
        {
            return new MPSNNReductionFeatureChannelsMeanNode (imageNode);
        }
    }

    public class SpatialMeanLayer : ReductionLayer
    {
        public override int[] GetOutputShape (params Tensor[] inputs)
        {
            var shape = inputs[0].Shape;
            var n = shape.Length;
            var r = new int[n];
            for (var i = 0; i < n - 1; i++) {
                r[i] = 1;
            }
            r[^1] = shape[^1];
            return r;
        }

        protected override MPSNNFilterNode CreateReductionNode (MPSNNImageNode imageNode)
        {
            return new MPSNNReductionSpatialMeanNode (imageNode);
        }
    }
}
