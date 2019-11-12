using System;
using System.Linq;
using System.Threading.Tasks;
using Foundation;
using MetalPerformanceShaders;
using MetalTensors.Tensors;

namespace MetalTensors.Layers
{
    public class AddLayer : Layer
    {
        public override int InputCount => 2;

        public override int[] GetOutputShape (params Tensor[] inputs)
        {
            return inputs[0].Shape;
        }

        protected override MPSNNFilterNode CreateFilterNode (MPSNNImageNode[] inputImageNodes)
        {
            return new MPSNNAdditionNode (inputImageNodes);
        }
    }
}
