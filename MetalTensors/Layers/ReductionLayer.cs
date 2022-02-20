using Metal;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public abstract class ReductionLayer : Layer
    {
        public override int MinInputCount => 1;

        public virtual bool IsSpatial => false;

        public override int[] GetOutputShape (params Tensor[] inputs)
        {
            var shape = inputs[0].Shape;
            var n = shape.Length;
            var r = new int[n];
            if (IsSpatial) {
                for (var i = 0; i < n - 1; i++) {
                    r[i] = 1;
                }
                r[^1] = shape[^1];
            }
            else {
                for (var i = 0; i < n - 1; i++) {
                    r[i] = shape[i];
                }
                r[^1] = 1;
            }
            return r;
        }

        protected override MPSNNFilterNode CreateFilterNode ((MPSNNImageNode ImageNode, int[] Shape)[] inputs, IMTLDevice device)
        {
            return CreateReductionNode (inputs[0].ImageNode);
        }

        protected abstract MPSNNFilterNode CreateReductionNode (MPSNNImageNode imageNode);
    }

    public class ArgMaxLayer : ReductionLayer
    {
        protected override MPSNNFilterNode CreateReductionNode (MPSNNImageNode imageNode) => new MPSNNReductionFeatureChannelsArgumentMaxNode (imageNode);
    }

    public class ArgMinLayer : ReductionLayer
    {
        protected override MPSNNFilterNode CreateReductionNode (MPSNNImageNode imageNode) => new MPSNNReductionFeatureChannelsArgumentMinNode (imageNode);
    }

    public class MaxLayer : ReductionLayer
    {
        protected override MPSNNFilterNode CreateReductionNode (MPSNNImageNode imageNode) => new MPSNNReductionFeatureChannelsMaxNode (imageNode);
    }

    public class MeanLayer : ReductionLayer
    {
        protected override MPSNNFilterNode CreateReductionNode (MPSNNImageNode imageNode) => new MPSNNReductionFeatureChannelsMeanNode (imageNode);
    }

    public class MinLayer : ReductionLayer
    {
        protected override MPSNNFilterNode CreateReductionNode (MPSNNImageNode imageNode) => new MPSNNReductionFeatureChannelsMinNode (imageNode);
    }

    public class SpatialMeanLayer : ReductionLayer
    {
        public override bool IsSpatial => true;
        protected override MPSNNFilterNode CreateReductionNode (MPSNNImageNode imageNode) => new MPSNNReductionSpatialMeanNode (imageNode);
    }

    public class SumLayer : ReductionLayer
    {
        protected override MPSNNFilterNode CreateReductionNode (MPSNNImageNode imageNode) => new MPSNNReductionFeatureChannelsSumNode (imageNode);
    }
}
