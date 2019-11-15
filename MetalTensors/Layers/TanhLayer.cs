using System;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class TanhLayer : UnopLayer
    {
        protected override MPSNNFilterNode CreateUnopNode (MPSNNImageNode imageNode)
        {
            return new MPSCnnNeuronTanHNode (imageNode);
        }
    }
}
