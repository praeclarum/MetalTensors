using System;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class SoftMaxLayer : UnopLayer
    {
        protected override MPSNNFilterNode CreateUnopNode (MPSNNImageNode imageNode)
        {
            return new MPSCnnSoftMaxNode (imageNode);
        }
    }
}
