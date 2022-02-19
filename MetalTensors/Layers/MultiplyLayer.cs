using System.Linq;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class MultiplyLayer : BinopLayer
    {
        public MultiplyLayer (string? name = null, bool isTrainable = true) : base (name, isTrainable: isTrainable)
        {
        }

        protected override MPSNNFilterNode CreateFilterNode ((MPSNNImageNode ImageNode, int[] Shape)[] inputs, IMTLDevice device)
        {
            return new MPSNNMultiplicationNode (inputs.Select (x => x.ImageNode).ToArray ());
        }
    }
}
