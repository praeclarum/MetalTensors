using System.Linq;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class MultiplyLayer : BinaryArithmeticLayer
    {
        protected override MPSNNFilterNode CreateFilterNode ((MPSNNImageNode ImageNode, int[] Shape)[] inputs, IMTLDevice device)
        {
            return new MPSNNMultiplicationNode (inputs.Select (x => x.ImageNode).ToArray ());
        }
    }
}
