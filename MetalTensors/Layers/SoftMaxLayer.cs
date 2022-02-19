using System;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class SoftMaxLayer : UnopLayer
    {
        public SoftMaxLayer (string? name = null, bool isTrainable = true) : base (name, isTrainable: isTrainable)
        {
        }

        protected override MPSNNFilterNode CreateUnopNode (MPSNNImageNode imageNode)
        {
            return new MPSCnnSoftMaxNode (imageNode);
        }
    }
}
