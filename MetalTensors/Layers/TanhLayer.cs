using System;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class TanhLayer : UnopLayer
    {
        public TanhLayer (string? name = null, bool isTrainable = true) : base (name, isTrainable: isTrainable)
        {
        }

        protected override MPSNNFilterNode CreateUnopNode (MPSNNImageNode imageNode)
        {
            return new MPSCnnNeuronTanHNode (imageNode);
        }
    }
}
