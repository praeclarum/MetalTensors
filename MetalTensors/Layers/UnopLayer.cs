using Metal;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public abstract class UnopLayer : Layer
    {
        public override int MinInputCount => 1;

        public override int[] GetOutputShape (params Tensor[] inputs) => inputs[0].Shape;

        protected override MPSNNFilterNode CreateFilterNode ((MPSNNImageNode ImageNode, int[] Shape)[] inputs, IMTLDevice device)
        {
            return CreateUnopNode (inputs[0].ImageNode);
        }

        protected abstract MPSNNFilterNode CreateUnopNode (MPSNNImageNode imageNode);
    }
}
