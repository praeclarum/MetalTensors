using System.Linq;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class DivideLayer : BinopLayer
    {
        public DivideLayer (string? name = null) : base (name)
        {
        }

        protected override MPSNNFilterNode CreateFilterNode ((MPSNNImageNode ImageNode, int[] Shape)[] inputs, IMTLDevice device)
        {
            return new MPSNNDivisionNode (inputs.Select (x => x.ImageNode).ToArray ());
        }
    }
}
