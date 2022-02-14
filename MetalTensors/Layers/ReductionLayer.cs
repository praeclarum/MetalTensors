using Metal;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public abstract class ReductionLayer : Layer
    {
        public override int MinInputCount => 1;

        protected override MPSNNFilterNode CreateFilterNode ((MPSNNImageNode ImageNode, int[] Shape)[] inputs, IMTLDevice device)
        {
            return CreateReductionNode (inputs[0].ImageNode);
        }

        protected abstract MPSNNFilterNode CreateReductionNode (MPSNNImageNode imageNode);
    }
}
