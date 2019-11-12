using System.Linq;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class SubtractLayer : BinaryArithmeticLayer
    {
        protected override MPSNNFilterNode CreateFilterNode ((MPSNNImageNode ImageNode, int[] Shape)[] inputs, IMTLDevice device)
        {
            return new MPSNNSubtractionNode (inputs.Select (x => x.ImageNode).ToArray ());
        }
    }
}
