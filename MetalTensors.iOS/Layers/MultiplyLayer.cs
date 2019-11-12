using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class MultiplyLayer : BinaryArithmeticLayer
    {
        protected override MPSNNFilterNode CreateFilterNode (MPSNNImageNode[] inputImageNodes)
        {
            return new MPSNNMultiplicationNode (inputImageNodes);
        }
    }
}
