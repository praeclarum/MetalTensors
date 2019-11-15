using System.Linq;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class AddLayer : BinopLayer
    {
        protected override MPSNNFilterNode CreateFilterNode ((MPSNNImageNode ImageNode, int[] Shape)[] inputs, IMTLDevice device)
        {
            return new MPSNNAdditionNode (inputs.Select (x => x.ImageNode).ToArray ());
        }
    }
}
