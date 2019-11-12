using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class DivideLayer : BinaryArithmeticLayer
    {
        protected override MPSNNFilterNode CreateFilterNode (MPSNNImageNode[] inputImageNodes)
        {
            return new MPSNNDivisionNode (inputImageNodes);
        }
    }
}
