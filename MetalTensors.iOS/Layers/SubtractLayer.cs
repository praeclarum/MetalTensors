using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class SubtractLayer : BinaryArithmeticLayer
    {
        protected override MPSNNFilterNode CreateFilterNode (MPSNNImageNode[] inputImageNodes)
        {
            return new MPSNNSubtractionNode (inputImageNodes);
        }
    }
}
